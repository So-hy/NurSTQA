from openai import OpenAI
import json
import argparse
import os
import openai
import tqdm
import time
import re

client = OpenAI()
openai.api_key = ""

def generate_sentences(prompt_template, table_info, model):
   

    table_caption = table_info.get('table_caption', '')
    table_columns = ', '.join(table_info.get('table_column_names', []))
    table_content = '\n'.join([', '.join(row) for row in table_info.get('table_content_values', [])])
    table_explain = table_info.get('text', '')
    
    prompt = prompt_template.replace('{{Table_Caption}}', table_caption)
    prompt = prompt.replace('{{Table_Column}}', table_columns)
    prompt = prompt.replace('{{Table_Content}}', table_content)
    prompt = prompt.replace('{{Table_Explain}}', table_explain)
    
    try:
        response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": prompt}],
        temperature=0,
        max_tokens=2000,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None
        )
        usage = response.usage
        print(usage)
        print(response.choices[0].message.content)
        return response.choices[0].message.content.strip(), usage
    
    except Exception as e:
        print(f"OpenAI API 오류: {e}")
        return None, None

def process_gpt_response(response):

    try:

        if response.startswith('```json') and response.endswith('```'):
            json_str = response.split('```json')[1].rstrip('```').strip()
        else:
            json_str = response.strip()
        
        # JSON 파싱
        qa_list = json.loads(json_str)
        return qa_list
    except json.JSONDecodeError as e:
        print(f"JSON 디코딩 오류: {e}")
        print(f"응답 내용: {response}")
        return []
    except Exception as e:
        print(f"응답 처리 오류: {e}")
        return []

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="GPT-4o mini를 사용하여 QA 쌍 생성.")
    parser.add_argument('--prompt_fp', type=str, default='prompt/qa_generation.txt')
    parser.add_argument('--save_fp', type=str, default='results/result2.json')
    parser.add_argument('--scigen_fp', type=str,default='data/dev.json')
    parser.add_argument('--model', type=str, default='gpt-4o-mini')
    args = parser.parse_args()
    
  
    with open(args.scigen_fp, 'r', encoding='utf-8') as f:
        scigen_data = json.load(f)
    

    with open(args.prompt_fp, 'r', encoding='utf-8') as f:
        prompt_template = f.read()
    
    results = {}
    token_counts = {} 
    ignore = 0
    max_retries = 3  


    for idx, (key, instance) in enumerate(tqdm.tqdm(scigen_data.items(), desc="Processing Tables")):
        table_id = idx 
        table_info = {
            'table_caption': instance.get('table_caption', ''),
            'table_column_names': instance.get('table_column_names', []),
            'table_content_values': instance.get('table_content_values', []),
            'text': instance.get('text', '')
        }


        if not any(table_info.values()):
            print(f"테이블 정보가 없음: {key}")
            ignore += 1
            continue


        retries = 0
        success = False
        while retries < max_retries and not success:
            try:
        
                gpt_response, usage = generate_sentences(prompt_template,table_info, args.model)
                if not gpt_response or not usage:
                    retries += 1
                    print(f"GPT 응답 없음. 테이블 {table_id}, 재시도 {retries}/{max_retries}")
                    time.sleep(2)
                    continue
    
                qa_list = process_gpt_response(gpt_response)
                if not qa_list:
                    retries += 1
                    print(f"QA 리스트 없음. 테이블 {table_id}, 재시도 {retries}/{max_retries}")
                    time.sleep(2)
                    continue
     
                results[str(table_id)] = qa_list
         
                token_counts[str(table_id)] = {
                    'input_token_count': usage.prompt_tokens if hasattr(usage, 'prompt_tokens') else 0,
                    'output_token_count': usage.completion_tokens if hasattr(usage, 'completion_tokens') else 0,
                    'total_token_count': usage.total_tokens if hasattr(usage, 'total_tokens') else 0,
                }
                success = True
  
                time.sleep(1)  
            except Exception as e:
                print(f"테이블 {table_id} 처리 오류: {e}")
                retries +=1
                time.sleep(2)
                continue
        if not success:
            ignore +=1
            print(f"테이블 {table_id} 처리 실패: {key}")

    print(f"Ignore: {ignore}")


    os.makedirs(os.path.dirname(args.save_fp), exist_ok=True)
    with open(args.save_fp, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)


    token_save_fp = 'results/token_count.json' 
    with open(token_save_fp, 'w', encoding='utf-8') as f:
        json.dump(token_counts, f, ensure_ascii=False, indent=4)