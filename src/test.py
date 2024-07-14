import json
import random
import numpy as np
from models import create_model
from utils import *
from attacker import Attacker
import torch
import datetime
from tqdm import tqdm

def main():
    args = parse_args()
    torch.cuda.set_device(args.gpu_id)
    device = 'cuda'
    setup_seeds(args.seed)
    if args.model_config_path == None:
        args.model_config_path = f'../config/{args.model_name}_config.json'

    llm = create_model(args.model_config_path)

    corpus, _, qrels = load_beir_datasets(args.eval_dataset, args.split)
    incorrect_answers = load_json(f'../middleDatas/incorrent_target/{args.eval_dataset}.json')
    random.shuffle(incorrect_answers)

    if args.origin_contriever_score is None: 
        args.origin_contriever_score = f"../middleDatas/contriever_score/{args.eval_dataset}-{args.eval_model_code}.json"
        
    with open(args.origin_contriever_score, 'r') as f:
        results = json.load(f)

    model, c_model, tokenizer, get_emb = load_models(args.eval_model_code)
    c_model.eval()
    c_model.to(device)
    attacker = Attacker(args)

    all_results = []
    asr_list = []
    ret_list = []

    current_time = datetime.datetime.now()
    time_string = current_time.strftime('%m%d%H%M')
    for iter in tqdm(range(args.batch), desc="Batch"):
        target_queries_idx = range(iter * args.batch_size, iter * args.batch_size + args.batch_size)
        target_queries = [incorrect_answers[idx]['question'] for idx in target_queries_idx]

        for i in target_queries_idx:
            target_queries[i - iter * args.batch_size] = {'query': target_queries[i - iter * args.batch_size], 'id': incorrect_answers[i]['id']}

        adv_text_groups = attacker.get_attack(target_queries)
        adv_text_list = sum(adv_text_groups, [])  # convert 2D array to 1D array

        adv_input = tokenizer(adv_text_list, padding=True, truncation=True, return_tensors="pt")
        adv_input = {key: value.cuda() for key, value in adv_input.items()}

        with torch.no_grad():
            adv_embs = get_emb(c_model, adv_input)

        asr_cnt = 0
        ret_sublist = []

        iter_results = []
        for i in tqdm(target_queries_idx, desc=f"Batch {iter+1}"):
            iter_idx = i - iter * args.batch_size  # iter index
            question = incorrect_answers[i]['question']
            incco_ans = incorrect_answers[i]['incorrect answer']

            topk_idx = list(results[incorrect_answers[i]['id']].keys())[:args.top_k]
            topk_results = [{'score': results[incorrect_answers[i]['id']][idx], 'context': corpus[idx]['text']} for idx in topk_idx]

            query_input = tokenizer(question, padding=True, truncation=True, return_tensors="pt")
            query_input = {key: value.cuda() for key, value in query_input.items()}
            with torch.no_grad():
                query_emb = get_emb(model, query_input)
            for j in range(args.adv_per_query):
                ind = args.adv_per_query* iter_idx + j
                adv_emb = adv_embs[ind, :].unsqueeze(0)
                adv_sim = torch.mm(adv_emb, query_emb.T).cpu().item()
                topk_results.append({'score': adv_sim, 'context': adv_text_list[ind]})

            topk_results = sorted(topk_results, key=lambda x: float(x['score']), reverse=True)
            topk_contents = [topk_results[j]["context"] for j in range(args.top_k)]
            # tracking the num of adv_text in topk
            adv_text_set = set(adv_text_groups[iter_idx])

            cnt_from_adv = sum([i in adv_text_set for i in topk_contents])
            ret_sublist.append(cnt_from_adv)

            query_prompt = wrap_prompt([question, topk_contents], prompt_id=3)
            response = llm.query(query_prompt)

            injected_adv = [i for i in topk_contents if i in adv_text_set]
            iter_results.append(
                {
                    "id": incorrect_answers[i]['id'],
                    "question": question,
                    "injected_adv": injected_adv,
                    "input_prompt": query_prompt,
                    "output_poison": response,
                    "incorrect_answer": incco_ans,
                    "answer": incorrect_answers[i]['correct answer']
                }
            )

            if clean_str(incco_ans) in clean_str(response):
                asr_cnt += 1

        asr_list.append(asr_cnt)
        ret_list.append(ret_sublist)

        all_results.append({f'iter_{iter}': iter_results})
        
        save_results(all_results, time_string, "debug"+str(iter))
        print(f'Saving iter results to ../results/query_results/{time_string}/debug{iter}.json')

    asr = np.array(asr_list) / args.batch_size
    asr_mean = round(np.mean(asr), 2)
    ret_precision_array = np.array(ret_list) / args.top_k
    ret_precision_mean = np.mean(ret_precision_array, axis=1)
    ret_recall_array = np.array(ret_list) / args.adv_per_query
    ret_recall_mean = np.mean(ret_recall_array, axis=1)

    ret_f1_array = f1_score(ret_precision_array, ret_recall_array)
    ret_f1_mean = np.mean(ret_f1_array, axis=1)

    # Save plots
    save_plot(asr, "Iterations", "ASR", "ASR Over Iterations", f"../results/{time_string}/ASR_results/ASR.png", asr_mean)
    
    for i in range(args.batch):
        save_plot(ret_precision_array[i], "Iterations", "Precision", f"Precision Over Iteration {i}", f"../results/{time_string}/Precision_results/Precision_{i}.png", np.mean(ret_precision_array[i]))
        save_plot(ret_recall_array[i], "Iterations", "Recall", f"Recall Over Iteration {i}", f"../results/{time_string}/Recall_results/Recall_{i}.png", np.mean(ret_recall_array[i]))
        save_plot(ret_f1_array[i], "Iterations", "F1 Score", f"F1 Score Over Iteration {i}", f"../results/{time_string}/F1_results/F1_{i}.png", np.mean(ret_f1_array[i]))

    save_plot(ret_precision_mean, "Iterations", "Precision", "Precision average Over Iterations", f"../results/{time_string}/Precision_results/PrecisionAvg.png")
    save_plot(ret_recall_mean, "Iterations", "Recall", "Recall average Over Iterations", f"../results/{time_string}/Recall_results/RecallAvg.png")
    save_plot(ret_f1_mean, "Iterations", "F1 Score", "F1 Score average Over Iterations", f"../results/{time_string}/F1_results/F1Avg.png")

if __name__ == '__main__':
    main()