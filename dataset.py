from transformers import AutoTokenizer
import transformers
import torch
import copy
from tqdm import tqdm
from torch.utils.data import Dataset
import json
import argparse


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
                
                # 指示文までの長さ
                source_len = source_tokenized['length'][0]
                
                # LLMに生成してほしい正解文章に指示文も含まれているので、
                # 指示文のところはCrossEntropyLossの損失を計算をしないようにIGNORE_INDEXとして-100で埋める
                labels[:source_len] = self.ignore_index
                
                self.features.append({
                    'input_ids': input_ids,
                    'labels': labels
                })
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx]


train_dataset = InstructDataset(args.json_file, tokenizer)
print(train_dataset[0])

