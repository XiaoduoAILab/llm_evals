import argparse
import json
import os
import random
import warnings

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import transformers
from tqdm import tqdm

from utils import get_task_prompt, get_task_data, get_prompt

MULTIPLE_CHOICE_TASKS = ['accountant',
                         'advanced_mathematics',
                         'art_studies',
                         'basic_medicine',
                         'business_administration',
                         'chinese_language_and_literature',
                         'civil_servant',
                         'clinical_medicine',
                         'college_chemistry',
                         'college_economics',
                         'college_physics',
                         'college_programming',
                         'computer_architecture',
                         'computer_network',
                         'discrete_mathematics',
                         'education_science',
                         'electrical_engineer',
                         'environmental_impact_assessment_engineer',
                         'fire_engineer',
                         'high_school_biology',
                         'high_school_chemistry',
                         'high_school_chinese',
                         'high_school_geography',
                         'high_school_history',
                         'high_school_mathematics',
                         'high_school_physics',
                         'high_school_politics',
                         'ideological_and_moral_cultivation',
                         'law',
                         'legal_professional',
                         'logic',
                         'mao_zedong_thought',
                         'marxism',
                         'metrology_engineer',
                         'middle_school_biology',
                         'middle_school_chemistry',
                         'middle_school_geography',
                         'middle_school_history',
                         'middle_school_mathematics',
                         'middle_school_physics',
                         'middle_school_politics',
                         'modern_chinese_history',
                         'operating_system',
                         'physician',
                         'plant_protection',
                         'probability_and_statistics',
                         'professional_tour_guide',
                         'sports_science',
                         'tax_accountant',
                         'teacher_qualification',
                         'urban_and_rural_planner',
                         'veterinary_medicine']


def setup_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


@torch.no_grad()
def generate_one_completion(prompt, model, tokenizer):
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(inputs, max_new_tokens=1, return_dict_in_generate=True, output_scores=True,
                             pad_token_id=tokenizer.eos_token_id)
    logits = outputs.scores[0][0]

    logits = torch.nn.functional.softmax(logits, dim=0, dtype=torch.float32)
    probs = torch.tensor(
        [
            logits[tokenizer("A").input_ids[0]],
            logits[tokenizer("B").input_ids[0]],
            logits[tokenizer("C").input_ids[0]],
            logits[tokenizer("D").input_ids[0]],
        ]
    )

    probs = probs.detach().cpu().numpy()
    probs = (probs)
    pred = {0: "A", 1: "B", 2: "C", 3: "D"}[np.argmax(probs)]

    return pred


def run_tasks(args, tasks, model, tokenizer):
    model_name = os.path.basename(os.path.dirname(args.model))
    # log_filename = 'execution_log.txt'
    log_filename = model_name + '.jsonl'

    if os.path.exists(log_filename):
        os.remove(log_filename)

    acc_total = 0
    num_total = 0

    with open(log_filename, 'w+') as f:
        for task in tqdm(tasks):
            acc = 0
            task_data = get_task_data(task)
            task_prompt = get_task_prompt(task, num_shot=5)

            num_examples = len(task_data)
            for i in range(num_examples):
                prompt = task_prompt
                q_ = task_data[i]
                prompt += get_prompt(q_)
                prompt += '答案：('

                if args.debug:
                    print('=======================================================')
                    print('prompt: ' + prompt)

                response = generate_one_completion(prompt, model, tokenizer)

                if args.debug:
                    print('>>>>>>>>>>>>>>>>>>>>')
                    print('Response: ' + response)

                ans_ = response

                if args.debug:
                    print('Model output: ' + ans_)

                a = q_['answer']

                if args.debug:
                    print('Answer: ' + a)

                correct = ans_ == a
                if args.debug:
                    print('correct: ' + str(correct))

                if correct:
                    acc += 1
                    acc_total += 1
                num_total += 1

                json_obj = dict(question=q_['question'], answer=a, output=ans_, correct=correct)
                f.write(json.dumps(json_obj))
                f.write('\n')
                f.flush()

            print('%s acc %.4f' % (task, acc / num_examples))

    return acc_total, num_total


def main(args, multiple_choice_tasks=MULTIPLE_CHOICE_TASKS):
    print(args)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model, skip_special_tokens=True, add_bos_token=False,
                                                           add_eos_token=False, clean_up_tokenization_spaces=True,
                                                           trust_remote_code=True, use_fast=args.use_fast)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    print(f"Loading model from {args.model}")

    model_config = transformers.AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    print(model_config)

    torch_dtype = model_config.torch_dtype
    if torch_dtype == torch.float32 or torch_dtype == torch.float64:
        torch_dtype = torch.float16

    model = transformers.AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch_dtype, device_map='auto',
                                                              trust_remote_code=True)

    model.eval()

    print(model)
    print(tokenizer)

    acc_total = 0
    num_total = 0

    acc, num = run_tasks(args, multiple_choice_tasks, model, tokenizer)
    acc_total += acc
    num_total += num
    print('%s acc %.4f' % ('C-EVAL avg', acc / num))


if __name__ == "__main__":
    setup_seeds(seed=42)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # To avoid warnings about parallelism in tokenizers
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, default="google/flan-t5-small")
    parser.add_argument("--prompts_dir", "-s", type=str, default="cot-prompts")
    parser.add_argument("--debug", type=bool, default=False)
    parser.add_argument('--num_beams', default=1, type=int)
    parser.add_argument('--seq_len', type=int, default=256, help='seq len')
    parser.add_argument('--use_fast', type=bool, default=False, help='use fast')

    args = parser.parse_args()

    main(args)
