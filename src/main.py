from models import create_model
from utils import *

def main():
    args = parse_args()
    if args.model_config_path == None:
        args.model_config_path = f'../config/{args.model_name}_config.json'

    llm = create_model(args.model_config_path)

    question = input("Please enter the question (default: What is the capital of France?): ") or "What is the capital of France?"
    incorrect_answer = input("Please enter the incorrect answer you want (default: Zhe Jiang): ") or "Zhe Jiang"
    nums = int(input("Please enter the number of adv_texts (default: 1): ") or 1)

    prompt = wrap_prompt([question,incorrect_answer],1)
    adv_texts = []
    for i in range(nums):
        response = llm.query(prompt)
        if 'Corpus:' in response:
            response = response.split('Corpus:')[1].strip()
        response = f"{question}. {response}"
        while response in adv_texts:
            response = llm.query(prompt)
            if 'Corpus:' in response:
                response = response.split('Corpus:')[1].strip()
            response = f"{question}. {response}"
        adv_texts.append(response)
    print("Your adv_texts:\n")
    for i in adv_texts:
        print(i)

if __name__ == '__main__':
    main()