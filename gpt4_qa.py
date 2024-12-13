from openai import OpenAI
import json
import openai
import argparse
import os
import tqdm
import time
from nltk.tokenize import sent_tokenize
import nltk
nltk.download('punkt_tab')

client = OpenAI()
openai.api_key = ""
def qa_sentences(prompt_template, table_data, question, model):

    table_section = (
        f"Table Caption: {table_data['table_caption']}\n"
        f"Table Headers: {', '.join(table_data['table_column_names'])}\n"
        f"Table Contents: {table_data['table_content_values']}"
    )

    prompt = prompt_template.replace("{{Table}}", table_section).replace("{{Question}}", question)
    

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": prompt}],
        temperature=0,
        max_tokens=50,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None
    )
    print(prompt)
    print(response.choices[0].message.content)
    return response.choices[0].message.content.strip()

if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--table_fp', type=str, default='data/nurst/nurst_table.json')
    argparser.add_argument('--qa_fp', type=str, default='data/nurst/nurst_test.json')
    argparser.add_argument('--prompt_fp', type=str, default='prompt/qa_task.txt')
    argparser.add_argument('--save_fp', type=str, default='results/gpt3.5_turbo_qa_results.json')
    argparser.add_argument('--model', type=str, default='gpt-3.5-turbo')
    args = argparser.parse_args()

    with open(args.table_fp, 'r', encoding='utf-8') as file:
        table_data = json.load(file)
    with open(args.qa_fp, 'r', encoding='utf-8') as file:
        qa_data = json.load(file)
    with open(args.prompt_fp, 'r', encoding='utf-8') as file:
        prompt_template = file.read()

    results = {}
    ignored = 0

    for table_id, table_content in tqdm.tqdm(table_data.items()):
        if table_id not in qa_data:
            print(f"Skipping ID {table_id}: No matching QA data")
            ignored += 1
            continue

        for qa in qa_data[table_id]:
            question = qa["Question"]
            try:
                answer = qa_sentences(
                    prompt_template,
                    table_content,
                    question,
                    args.model
                )


                results.setdefault(table_id, []).append({
                    "Question": question,
                    "Generated Answer": answer,
                    "Expected Answer": qa["Answer"],
                    "Tag": qa["Tag"]
                })
            except Exception as e:
                print(f"테이블 {table_id} 처리 오류: {e}, question: {question}\n{e}")
                ignored += 1


    with open(args.save_fp, 'w', encoding='utf-8') as file:
        json.dump(results, file, indent=4, ensure_ascii=False)

    print(f"Ignore: {ignored}")