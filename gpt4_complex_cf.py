from openai import OpenAI
import json
import argparse
import os
import tqdm
import time
import openai


client = OpenAI()
openai.api_key = ""

def generate_sentences(prompt_template, table_info, qa_data, model):
  
    try:
   
        table_caption = table_info.get('table_caption', 'N/A')
        table_columns = ', '.join(table_info.get('table_column_names', []))
        table_content = '\n'.join([', '.join(row) for row in table_info.get('table_content_values', [])])
        table_explain = table_info.get('text', '')

    
        qa_section = ""
        for idx, qa in enumerate(qa_data):
            qa_section += f"""
                {idx + 1}. Question: {qa['Question']}
                Answer: {qa['Answer']}
                Tag: {qa['Tag']}
                Explanation: {qa['Explanation']}
                """

 
        prompt = prompt_template.replace('{{Table_Caption}}', table_caption)
        prompt = prompt.replace('{{Table_Column}}', table_columns)
        prompt = prompt.replace('{{Table_Content}}', table_content)
        prompt = prompt.replace('{{Table_Explain}}', table_explain)
        prompt = prompt.replace('{{QA_Data}}', qa_section)

        print(prompt)

        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": prompt}],
            temperature=0,
            max_tokens=2000,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        usage = response.usage
        print(response.choices[0].message.content)
        return response.choices[0].message.content.strip(), usage

    except Exception as e:
        print(e)
        return None, None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="GPT-based QA Evaluation")
    parser.add_argument('--prompt_fp', type=str, default='prompt/complex_qa_classification.txt')
    parser.add_argument('--save_fp', type=str, default='results/complex_cf_result.json')
    parser.add_argument('--dev_fp', type=str, default='data/dev.json')
    parser.add_argument('--qa_fp', type=str, default='results/complex_result.json')
    parser.add_argument('--model', type=str, default='gpt-4o-mini')
    args = parser.parse_args()

 
    with open(args.dev_fp, 'r', encoding='utf-8') as f:
        table_data = json.load(f)
    with open(args.qa_fp, 'r', encoding='utf-8') as f:
        qa_data = json.load(f)


    with open(args.prompt_fp, 'r', encoding='utf-8') as f:
        prompt_template = f.read()

    results = {}
    token_counts = {}
    ignored_count = 0
    max_retries = 3


    for idx, (key, instance) in enumerate(tqdm.tqdm(table_data.items(), desc="Processing Tables")):
        table_id = idx

        table_info = {
            'table_caption': instance.get('table_caption', ''),
            'table_column_names': instance.get('table_column_names', []),
            'table_content_values': instance.get('table_content_values', []),
            'text': instance.get('text', ''),
        }


        qa_list = qa_data.get(str(table_id), [])
        if not qa_list:
            print(f"테이블 ID {str(table_id)}에 대한 QA 데이터 없음. 스킵.")
            ignored_count += 1
            continue

        retries = 0
        success = False
        while retries < max_retries and not success:
            try:
 
                gpt_response, usage = generate_sentences(prompt_template, table_info, qa_list, args.model)
                if not gpt_response or not usage:
                    retries += 1
                    print(f"GPT 응답 없음. 테이블 {table_id}, 재시도 {retries}/{max_retries}")
                    time.sleep(2)
                    continue

      
                results[table_id] = gpt_response


                token_counts[table_id] = {
                    'input_tokens': usage.prompt_tokens if hasattr(usage, 'prompt_tokens') else 0,
                    'output_tokens': usage.completion_tokens if hasattr(usage, 'completion_tokens') else 0,
                    'total_tokens': usage.total_tokens if hasattr(usage, 'total_tokens') else 0
                }
                success = True

            except Exception as e:
                retries += 1
                print(f"테이블 {table_id} 처리 오류: {e}, 재시도 {retries}/{max_retries}")
                time.sleep(2)

        if not success:
            ignored_count += 1
            print(f"테이블 {table_id} 처리 실패")


    os.makedirs(os.path.dirname(args.save_fp), exist_ok=True)
    with open(args.save_fp, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)


    token_counts_fp = args.save_fp.replace('.json', '_token_counts.json')
    with open(token_counts_fp, 'w', encoding='utf-8') as f:
        json.dump(token_counts, f, ensure_ascii=False, indent=4)