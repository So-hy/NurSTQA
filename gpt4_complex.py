import json
import argparse
import os
import openai
import tqdm
import time
import random
from openai import OpenAI


client = OpenAI()
openai.api_key = ""

def generate_sentences(prompt_template, table_info, model):

    table_caption = table_info.get('table_caption', '')
    table_columns = ', '.join(table_info.get('table_column_names', []))
    table_content = '\n'.join([', '.join(row) for row in table_info.get('table_content_values', [])])
    table_explain = table_info.get('text', '')
    

    combination_instructions = ''
    for idx, combo in enumerate(table_info.get('combination_instructions', []), 1):
        combination_instructions += f"{idx}. **[{combo[0]}, {combo[1]}]**\n"


    prompt = prompt_template.replace('{{Table_Caption}}', table_caption)
    prompt = prompt.replace('{{Table_Column}}', table_columns)
    prompt = prompt.replace('{{Table_Content}}', table_content)
    prompt = prompt.replace('{{Table_Explain}}', table_explain)
    prompt = prompt.replace('{{combination_instructions}}', combination_instructions)


    try:
        response =  client.chat.completions.create(
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
        print(e)
        return None, None

def process_gpt_response(response):

    try:
        qa_list = json.loads(response)
        return qa_list
    except json.JSONDecodeError as e:
        print(f"JSON decoding error: {e}")
        print(f"Response content: {response}")
        return []
    except Exception as e:
        print(f"Response processing error: {e}")
        return []

if __name__ == '__main__':
    # Argument parsing
    parser = argparse.ArgumentParser(description="Generate QA pairs using OpenAI GPT.")
    parser.add_argument('--prompt_fp', type=str, default='prompt/complex_qa_generation.txt', help='Path to the prompt template file.')
    parser.add_argument('--save_fp', type=str, default='results/complex_result.json', help='Path to save the result file.')
    parser.add_argument('--scigen_fp', type=str, default='data/dev.json', help='Path to the dataset JSON file.')
    parser.add_argument('--model', type=str, default='gpt-4o-mini', help='OpenAI model name to use (e.g., "gpt-4").')
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
            'text': instance.get('text', ''),
        }

        valid_combinations = [[1, 2], [1, 3], [2, 3], [3, 4], [2, 4]]
        selected_combinations = random.sample(valid_combinations, 3)
        table_info['combination_instructions'] = selected_combinations

        if not any(table_info.values()):
            print(f"테이블 ID {key}에 대한 QA 데이터 없음. 스킵.")
            ignore += 1
            continue

        retries = 0
        success = False
        while retries < max_retries and not success:
            try:

                gpt_response, usage = generate_sentences(prompt_template, table_info, args.model)
                if not gpt_response or not usage:
                    retries += 1
                    print(f"테이블 {table_id} 처리 오류: {e}, 재시도 {retries}/{max_retries}")
                    continue

                qa_list = process_gpt_response(gpt_response)
                if not qa_list:
                    retries += 1
                    print(f"테이블 {table_id} 처리 실패")
  
                    continue
          
                results[str(table_id)] = qa_list
          
                token_counts[str(table_id)] = {
                    'input_token_count': usage.prompt_tokens if hasattr(usage, 'prompt_tokens') else 0,
                    'output_token_count': usage.completion_tokens if hasattr(usage, 'completion_tokens') else 0,
                    'total_token_count': usage.total_tokens if hasattr(usage, 'total_tokens') else 0,
                }
                success = True

            except Exception as e:
                print(f"테이블 {table_id} 처리 오류: {e}")
                retries += 1

                continue
        if not success:
            ignore += 1
            print(f"테이블 {table_id} 처리 실패")

    print(f"Ignore: {ignore}")


    os.makedirs(os.path.dirname(args.save_fp), exist_ok=True)
    with open(args.save_fp, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)


    token_save_fp = 'results/complex_token_count.json'
    with open(token_save_fp, 'w', encoding='utf-8') as f:
        json.dump(token_counts, f, ensure_ascii=False, indent=4)