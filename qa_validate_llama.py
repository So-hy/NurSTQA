import json
import argparse
import os
import tqdm
import time
from transformers import AutoProcessor, AutoModelForImageTextToText
import torch
import re
from nltk.tokenize import sent_tokenize
import nltk
nltk.download('punkt')


HUGGINGFACE_ACCESS_TOKEN = "" 

processor = AutoProcessor.from_pretrained(
    "meta-llama/Llama-3.2-11B-Vision-Instruct",
    use_auth_token=HUGGINGFACE_ACCESS_TOKEN
)
model = AutoModelForImageTextToText.from_pretrained(
    "meta-llama/Llama-3.2-11B-Vision-Instruct",
    use_auth_token=HUGGINGFACE_ACCESS_TOKEN,
    device_map="auto",
    torch_dtype=torch.float16
)

def load_prompt_template(prompt_fp):

    with open(prompt_fp, 'r', encoding='utf-8') as f:
        return f.read()

def generate_validation_prompt(template, table_info, qa):

    prompt = template
    prompt = prompt.replace("{{Table_Caption}}", table_info.get('table_caption', ''))
    prompt = prompt.replace("{{Table_Column}}", ', '.join(table_info.get('table_column_names', [])))
    prompt = prompt.replace("{{Table_Content}}", '\n'.join([', '.join(row) for row in table_info.get('table_content_values', [])]))
    prompt = prompt.replace("{{Table_Explain}}", table_info.get('text', ''))
    prompt = prompt.replace("{{Tag}}", qa.get('Tag', ''))
    prompt = prompt.replace("{{Question}}", qa.get('Question', ''))
    prompt = prompt.replace("{{Answer}}", qa.get('Answer', ''))
    prompt = prompt.replace("{{Explanation}}", qa.get('Explanation', ''))
    return prompt

def validate_qa_with_llama(table_info, qa, prompt_template, response_dict, index):

    prompt = generate_validation_prompt(prompt_template, table_info, qa)
  
    try:
        inputs = processor(text=prompt, return_tensors="pt").to(model.device)
        output = model.generate(
            **inputs,
            max_new_tokens=150,     
            temperature=0.01,        
            top_p=1.0,
            repetition_penalty=1.2
        )

        response = processor.batch_decode(output, skip_special_tokens=True)[0]
        response_dict[str(index)] = response

    except Exception as e:
        print(f"Llama API 오류: {e}")
        response_dict[str(index)] = f"ERROR: {str(e)}"

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Llama 모델을 사용하여 QA 쌍 검증.")
    parser.add_argument('--prompt_fp', type=str, default='prompt/qa_validate.txt', help='검증 프롬프트 파일 경로.')
    parser.add_argument('--save_fp', type=str, default='results/validation_report.json', help='검증 보고서를 저장할 파일 경로.')
    parser.add_argument('--result_fp', type=str, default='results/result.json', help='QA 쌍이 저장된 JSON 파일 경로.')
    parser.add_argument('--data_fp', type=str, default='data/dev.json', help='테이블 데이터가 저장된 JSON 파일 경로.')
    args = parser.parse_args()


    try:
        prompt_template = load_prompt_template(args.prompt_fp)
    except Exception as e:
        print(f"프롬프트 템플릿 로드 오류: {e}")
        exit(1)

    try:
        result_data = load_json(args.result_fp) 
    except Exception as e:
        print(f"QA 쌍 데이터 로드 오류: {e}")
        exit(1)


    try:
        table_data = load_json(args.data_fp)
    except Exception as e:
        print(f"테이블 데이터 로드 오류: {e}")
        exit(1)

    raw_responses = {}
    index = 0
    for table_id, qa_list in tqdm.tqdm(result_data.items(), desc="Validating QA Pairs with Llama"):
        table_info = table_data.get(table_id, {})
        if not table_info:
            print(f"테이블 ID {table_id}에 대한 테이블 데이터가 존재하지 않습니다.")
            continue

        raw_responses[table_id] = {}
        for qa in qa_list:
            validate_qa_with_llama(table_info, qa, prompt_template, raw_responses[table_id], index)
            index += 1
    
        if index == 50:
            break

    raw_output_path = 'results/raw_responses.json'
    save_json(raw_responses, raw_output_path)

