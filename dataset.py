from transformers import AutoTokenizer
import transformers
import torch
import copy
from tqdm import tqdm
from torch.utils.data import Dataset
import json
import argparse
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader


parser = argparse.ArgumentParser()
parser.add_argument("json_file", help="json file to be scored")
args = parser.parse_args()


# 初期設定
# GPUが使えるか確認
model = "meta-llama/Llama-2-7b-chat-hf"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(torch.cuda.get_device_name())
print("使用デバイス:", device)


tokenizer = AutoTokenizer.from_pretrained(model, token=True)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
    

)


tokenizer.pad_token = tokenizer.eos_token

class InstructDataset(Dataset):
    def __init__(self, json_data, tokenizer, ignore_index=-100):
        self.tokenizer = tokenizer
        self.ignore_index = ignore_index
        self.features = []
        with open(json_data) as fi:
            json_data = json.load(fi)
            for j in tqdm(json_data):            
                # 文末にEOSトークンを挿入
                example_text = j['input'] + self.tokenizer.eos_token
                
                # 指示文と回答文を全てtokenize
                example_tokenized = self.tokenizer(
                    example_text, 
                    padding='longest', 
                    truncation=True, 
                    max_length=512, 
                    return_tensors='pt'
                )
                
                input_ids = example_tokenized['input_ids'][0]
                
                # LLMが生成してほしい正解の文章として入力文をそのままコピーする
                labels = copy.deepcopy(input_ids)
                
                self.features.append({
                    'input_ids': input_ids,
                    'labels': labels
                })
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx]


train_dataset = InstructDataset(args.json_file, tokenizer)
#print(train_dataset[0])

class InstructCollator():
    def __init__(self, tokenizer, ignore_index=-100):
        self.tokenizer = tokenizer
        self.ignore_index = -100

    def __call__(self, examples):
        input_batch = []
        label_batch = []
        for example in examples:
            input_batch.append(example['input_ids'])
            label_batch.append(example['labels'])
        
        input_ids = pad_sequence(
            input_batch, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )

        # labelsのpaddingトークンは先程と同様にignore_indexである-100で埋める
        labels = pad_sequence(
            label_batch, batch_first=True, padding_value=self.ignore_index
        )

        # attention_maskはbool値でもいいらしい
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
            
        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask
        }


collator = InstructCollator(tokenizer)
loader = DataLoader(train_dataset, collate_fn=collator, batch_size=8, shuffle=True)
batch = next(iter(loader))
print(batch)