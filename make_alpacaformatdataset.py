import json
import argparse
import random


system_content = '''Your task is to classify the labels corresponding to the grading items from the input answer. Please refer to the Classification Rubric when performing the task.
_Your Outputs_
G2-3: _Your Outputs_
Justification Cue: _Your Outputs_

_Prompt_
日曜日は別ですが、父はふだん朝6時に起きています。

_Grading item_
G2-3:Word form of the expression corresponding to  "起きています"

_Classification Rubric_
G2-3: 2__The third person singular form of the word
G2-3: 0__Not in the third person singular.

'''


def input_prompt(instruction, input_ans):
    if input != "":
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
        # Instruction:
        {instruction}
        
        # Input:
        {input_ans}
        """
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

# Instruction:
{instruction}
"""


def output_prompt(output):
    if input != "":
        return f"""
        # Response:
        {output}
        <|endoftext|>
        """
    else:
        return f"""

# Response:
{output}
<|endoftext|>
"""



def main():
    args = parse_args()
    json_train_data = args.json_data
    num = args.pick_num
    item = args.score_item
    divi = args.divi_num
    qnum = args.question_num

    

    output_jsonl_name = "finetune-train" + qnum + item + 'divi' + str(divi)  + '.json'

    with open(json_train_data, 'r') as fi:
        data = json.load(fi)
        item_name = item + "_Score"
        print(item_name)
        ans_data = []
        score_data = []
        some_list = []
        json_list = []

        
        #  ○
        for dic in dcit_random_get(data, item_name, "2", num, seed=0):
            ans = dic["response"]
            label_cue = f'{item}:{dic[item_name]}\nJustification Cue:{num2word(dic[item],dic["response"])}'
            ans_data.append(ans)
            score_data.append(label_cue)
            
        for dic in dcit_random_get(data, item_name, "0", num, seed=0):
            ans_dic = dic["response"]
            label_cue = f'{item}:{dic[item_name]}\nJustification Cue:{num2word(dic[item],dic["response"])}'
            ans_data.append(ans_dic)
            score_data.append(label_cue)

    with open(output_jsonl_name, mode='w', encoding='utf-8') as fout:
        for obj, sco in zip(ans_data, score_data):
            dict_obj = {'input':input_prompt(system_content, obj),'output':output_prompt(sco)}
            json_list.append(dict_obj)
        json.dump(json_list, fout, ensure_ascii=False)
            #fout.write('\n')



#  trainデータからfewshot用の例をラベルごとに抜き出す関数
def dcit_random_get(data, key_to_find, value_to_find, sample_size,seed=None):
    filtered_data = []
    selected_dicts = []
    #  特定のキーとバリューを指定
    filtered_data = [dic for dic in data if dic.get(key_to_find) == int(value_to_find)]
    #  サンプル取得数を指定
    num_samples = min(sample_size, len(filtered_data))
    #  シード値を設定
    if seed is not None:
        random.seed(seed)
    #  取得
    selected_dicts = random.sample(filtered_data, num_samples) if filtered_data else []
    
    
    return selected_dicts


#　　根拠箇所を抜き出す関数
def num2word(num_str, word_str):
    num_list = num_str.split()
    #print(num_list)
    word_list = word_str.split()
    #print(word_list)
    selected_words = [word for number, word in zip(num_list, word_list) if int(number) == 1]
    
    return ' '.join(selected_words)


#  引数の指定
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-jd", "--json_data", dest="json_data", type=str,
                        metavar='<str>', required=True, help="The path to the data (.json)")
    parser.add_argument("-item", "--score_item", dest="score_item", type=str,
                        metavar='<str>', required=True, help="score_item") 
    parser.add_argument("-qnum", "--question_num", dest="question_num", type=str,
                        metavar='<str>', required=True, help="question_num") 
    parser.add_argument("-divi", "--divi_num", dest="divi_num", type=int,
                        metavar='<str>', required=True, help="divi_num")
    parser.add_argument("-num", "--pick_num", dest="pick_num", type=int,
                        metavar='<str>', required=True, help="pick_num")         
    args = parser.parse_args()
    return args



#  main
if __name__ == '__main__':
    main()
