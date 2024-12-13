import json
import argparse
import os
import openai
import tqdm
from openai import OpenAI

client = OpenAI()
openai.api_key = ""

def evaluate_answer(prompt_template, question, generated_answer, expected_answer, model):

    generated_answer = str(generated_answer)
    expected_answer = str(expected_answer)
    question = str(question)

    prompt = prompt_template.replace("{{Question}}", question)
    prompt = prompt.replace("{{Generated Answer}}", generated_answer)
    prompt = prompt.replace("{{Expected Answer}}", expected_answer)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": prompt}],
            temperature=0,
            max_tokens=200,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None
        )
        print("Prompt",prompt)
        print("Response: ",response.choices[0].message.content)
        usage = response.usage
        return response.choices[0].message.content.strip(), usage
    except Exception as e:
        print(f"OpenAI API error: {e}")
        return None, None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate Generated Answers against Expected Answers.")
    parser.add_argument('--data_fp', type=str, default='qa_results/gpt3.5_turbo_qa_results.json', help='Path to the data JSON file.')
    parser.add_argument('--prompt_fp', type=str, default='prompt/gpt_validate.txt', help='Path to the prompt template file.')
    parser.add_argument('--save_fp', type=str, default='qa_results/gpt_eval_easy/gpt3.5_turbo_results.json', help='Path to save the evaluation results.')
    parser.add_argument('--model', type=str, default='gpt-4o-mini', help='OpenAI model name to use (e.g., "gpt-4").')
    args = parser.parse_args()

    with open(args.data_fp, 'r', encoding='utf-8') as f:
        data = json.load(f)

    with open(args.prompt_fp, 'r', encoding='utf-8') as f:
        prompt_template = f.read()

    results = {}
    token_counts = {}
    ignore = 0
    max_retries = 3

    for key, items in tqdm.tqdm(data.items(), desc="Processing Items"):
        results[key] = []
        for idx, item in enumerate(items):
            question = item.get("Question", "")
            generated_answer = item.get("Generated Answer", "")
            expected_answer = item.get("Expected Answer", "")

            retries = 0
            success = False
            while retries < max_retries and not success:
                try:
                    evaluation, usage = evaluate_answer(
                        prompt_template, question, generated_answer, expected_answer, args.model
                    )
                    if not evaluation or not usage:
                        retries += 1
                        print(f"데이터 {idx}, {key} 처리 오류. 재시도 {retries}/{max_retries}")
                        continue

            
                    results[key].append({
                        "Question": question,
                        "Generated Answer": generated_answer,
                        "Expected Answer": expected_answer,
                        "Evaluation": evaluation
                    })

       
                    token_counts[f"{key}_{idx}"] = {
                        "input_tokens": usage.prompt_tokens if hasattr(usage, 'prompt_tokens') else 0,
                        "output_tokens": usage.completion_tokens if hasattr(usage, 'completion_tokens') else 0,
                        "total_tokens": usage.total_tokens if hasattr(usage, 'total_tokens') else 0
                    }
                    success = True
                except Exception as e:
                    print(f"평가 오류 {idx}, {key}: {e}")
                    retries += 1

            if not success:
                ignore += 1
                print(f"처리 실패 {idx}, {key}")

    print(f"Ignore: {ignore}")

  
    os.makedirs(os.path.dirname(args.save_fp), exist_ok=True)
    with open(args.save_fp, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    
    token_save_fp = args.save_fp.replace('.json', '_tokens.json')
    with open(token_save_fp, 'w', encoding='utf-8') as f:
        json.dump(token_counts, f, ensure_ascii=False, indent=4)
