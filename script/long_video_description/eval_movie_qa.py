import argparse
import time
import requests
import json

def load_prompt(file):
    prompt = ''
    with open(file) as fin:
        for line in fin:
            prompt = prompt + line
    return prompt

class AzureClient:
    def __init__(self, api_key, repeat_num=50):
        self.url = api_key
        self.repeat_num = repeat_num
    
    def __call__(self, prompt, model='gpt-4', max_tokens=1000):
        for _ in range(self.repeat_num):
            try:
                resp = requests.post(
                    self.url,
                    headers={'Content-Type': 'application/json'},
                    json={
                        'model': model,
                        'messages': [{
                            'role': 'user',
                            'content': prompt,
                        }],
                        'max_tokens': max_tokens,
                    },
                )
                rst = resp.json()['choices'][0]['message']['content']
                if model =='gpt-4':
                    pass
                else:
                    pass
                return rst
            except:
                time.sleep(1)
                continue
        
        return ''
call_search_azure = AzureClient("Your-Azure-Key")

def call_model(prompt):
    for _ in range(3):
    # while True:
        rst = call_search_azure(prompt=prompt, model='gpt-4-0613')
        if len(rst) <= 3:
            time.sleep(10)
        else:
            return rst
    return "[Answer]: A"

def get_answer_label(s):
    token = '[Answer]'
    s = s[s.find(token) + len(token) + 1:]
    if len(s) > 5:
        return 4
    if 'a' in s.lower():
        return 0
    elif 'b' in s.lower():
        return 1
    elif 'c' in s.lower():
        return 2
    elif 'd' in s.lower():
        return 3
    return 4

def get_model_data(input_file, movie_id_key, movie_description_key):
    model_data = dict()
    with open(input_file) as fin:
        for line in fin:
            data = json.loads(line)
            model_data[data[movie_id_key]] = data[movie_description_key]
    return model_data

def eval(model_name, model_data, output_file):
    tot = 0
    acc = 0
    with open('data/raw_data/movie_qa.jsonl') as fin, open(output_file, 'w') as fout:
        for line in fin:
            data = json.loads(line)
            mid = data['id']
            if mid not in model_data:
                print('movie_id missing: ' + str(mid))
                continue
            movie_description = model_data[mid]
            data["pred_movie_description"] = movie_description
            for qa in data['qa_list']:
                question = qa['question'] + '\nA. ' + qa['option_0'] + '\nB. ' + qa['option_1'] + '\nC. ' + qa['option_2'] + '\nD. ' + qa['option_3'] + "\nE. I don't know."
                answer = qa['answer']

                prompt = load_prompt('script/eval_prompt').format(text = movie_description, question=question)
                model_answer = call_model(prompt)
                model_answer_label = get_answer_label(model_answer)
                if model_answer_label == answer:
                    model_answer_evaluation = 1
                    acc += 1
                else:
                    model_answer_evaluation = 0
                tot += 1

                qa[model_name + '_response'] = model_answer
                qa[model_name + '_answer'] = model_answer_label
                qa[model_name + '_answer_evaluation'] = model_answer_evaluation
            fout.write(json.dumps(data, ensure_ascii=False) + '\n')
            fout.flush()

    print(acc / tot)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_output_path', type=str)
    parser.add_argument('--eval_output_path', type=str)
    parser.add_argument('--movie_id_key', type=str)
    parser.add_argument('--movie_description_key', type=str)
    parser.add_argument('--model_name', type=str, default='model')
    args = parser.parse_args()
    
    model_data = get_model_data(args.model_output_path, args.movie_id_key, args.movie_description_key)
    eval(args.model_name, model_data, args.eval_output_path)
