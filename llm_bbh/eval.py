import argparse
import json
import os
import random
import warnings

import numpy as np
import torch
import transformers
from tqdm import tqdm
from transformers import StoppingCriteriaList

MULTIPLE_CHOICE_TASKS = [
    'temporal_sequences', 'disambiguation_qa', 'date_understanding', 'tracking_shuffled_objects_three_objects',
    'penguins_in_a_table',
    'geometric_shapes', 'snarks', 'ruin_names', 'tracking_shuffled_objects_seven_objects',
    'tracking_shuffled_objects_five_objects',
    'logical_deduction_three_objects', 'hyperbaton', 'logical_deduction_five_objects',
    'logical_deduction_seven_objects', 'movie_recommendation',
    'salient_translation_error_detection', 'reasoning_about_colored_objects',
]
FREE_FORM_TASKS = [
    'multistep_arithmetic_two', 'navigate', 'dyck_languages', 'word_sorting', 'sports_understanding',
    'boolean_expressions', 'object_counting', 'formal_fallacies', 'causal_judgement', 'web_of_lies',
]


def custom_stopping_criteria(input_ids: torch.LongTensor, score: torch.FloatTensor, **kwargs) -> bool:
    # last_token = input_ids[0][-1]
    # for stop in stop_words_ids:
    #     if tokenizer.decode(stop) == tokenizer.decode(last_token):
    #         return True
    if input_ids[0][-1] == 13 and input_ids[0][-2] == 13:
        return True

    return False


@torch.no_grad()
def generate_one_completion(prompt, model, tokenizer):
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    num_tokens = inputs.shape[-1]
    outputs = model.generate(inputs, max_new_tokens=args.seq_len, pad_token_id=tokenizer.eos_token_id,
                             stopping_criteria=StoppingCriteriaList([custom_stopping_criteria]))
    text = tokenizer.decode(outputs[0][num_tokens:]).strip()
    return text


def extract_ans(ans, mode):
    ans_line = ans.split('answer is ')
    # Expect to see 'answer is'. If not return ""
    if len(ans_line) == 1:
        return ans
    else:
        ans = ans_line[-1].strip()

    if mode == 'multiple_choice':
        options = ['(A)', '(B)', '(C)', '(D)', '(E)', '(F)', '(G)', '(H)', '(I)', '(J)', '(K)', '(L)', '(M)', '(N)',
                   '(O)', '(P)', '(Q)', '(R)', '(S)', '(T)', '(U)', '(V)', '(W)', '(X)', '(Y)', '(Z)']
        # for option in options:
        #     if option in ans:
        #         ans = option[1]
        #         break
        # return ans

        positions = []
        for opt in options:
            if opt in ans:
                positions.append(ans.index(opt))
            else:
                positions.append(float('inf'))

        if positions:
            return options[np.argmin(positions)][1]
        else:
            return 'Z'

    elif mode == 'free_form':
        if '.' in ans:
            pos = ans.index('.')
            ans = ans[:pos]

        return ans


def run_tasks(args, tasks, model, tokenizer, mode):
    model_name = os.path.basename(os.path.dirname(args.model))
    logname = model_name + '.txt'
    if os.path.exists(logname):
        os.remove(logname)

    acc_total = 0
    num_total = 0

    with open(logname, 'w') as fd:
        for task in tqdm(tasks):
            # print('Testing %s ...' % task)
            fd.write('Testing {} ...'.format(task))

            acc = 0
            task_data = json.load(open('bbh/%s.json' % task))
            task_prompt = open('cot-prompts/%s.txt' % task, 'r').read()
            task_prompt = task_prompt[111:]

            num_examples = len(task_data['examples'])
            for i in range(num_examples):
                q_ = task_data['examples'][i]
                q = '\n\nQ: ' + q_['input']

                prompt_q = task_prompt + q + "\nA: Let's think step by step."
                prompt = "Follow the given examples and answer the question.\n\n" + prompt_q

                if args.debug:
                    print('=======================================================')
                    print('prompt: ' + prompt)

                response = generate_one_completion(prompt, model, tokenizer)
                ans_model = response

                if args.debug:
                    print('>>>>>>>>>>>>>>>>>>>>')
                    print('Response: ' + response)

                ans_ = extract_ans(response, mode)

                if args.debug:
                    print('Model output: ' + ans_)

                if mode == 'multiple_choice':
                    a = q_['target'][1]
                elif mode == 'free_form':
                    a = q_['target']

                if args.debug:
                    print('Answer: ' + a)

                fd.write('%s\nA_model:\n%s\nA_target:\n%s\n\n' % (q, ans_model, a))

                correct = ans_ == a
                if correct:
                    acc += 1
                    acc_total += 1
                num_total += 1

                log_line = json.dumps(dict(question=q_['input'], answer=a, output=ans_, correct=correct))
                if args.debug:
                    print('PROBLEM {}/{} {}'.format(i, num_examples, log_line))

            print('%s acc %.4f' % (task, acc / num_examples))

    return acc_total, num_total


def main(args, multiple_choice_tasks=MULTIPLE_CHOICE_TASKS, free_form_tasks=FREE_FORM_TASKS):
    print(args)

    model_config = transformers.AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    print(model_config)

    torch_dtype = model_config.torch_dtype
    if torch_dtype == torch.float32 or torch_dtype == torch.float64:
        torch_dtype = torch.float16

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model, skip_special_tokens=True, add_bos_token=False,
                                                           add_eos_token=False, clean_up_tokenization_spaces=True,
                                                           use_fast=False,
                                                           trust_remote_code=True)

    print(f"Loading model from {args.model}")
    model = transformers.AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch_dtype, device_map='auto',
                                                              trust_remote_code=True)
    model.eval()

    print(model)
    print(tokenizer)

    acc_total = 0
    num_total = 0

    run_multiple_choice = args.task == 'all' or args.task == 'multiple_choice'
    run_free_form = args.task == 'all' or args.task == 'free_form'

    if run_multiple_choice:
        acc, num = run_tasks(args, multiple_choice_tasks, model, tokenizer, mode='multiple_choice')
        acc_total += acc
        num_total += num
        print('%s acc %.4f' % ('multiple_choice_tasks', acc / num))
    if run_free_form:
        acc, num = run_tasks(args, free_form_tasks, model, tokenizer, mode='free_form')
        acc_total += acc
        num_total += num
        print('%s acc %.4f' % ('free_form_tasks', acc / num))

    print('%s acc %.4f' % ('total', acc_total / num_total))
    return


def seed_everything(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    seed_everything(42)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # To avoid warnings about parallelism in tokenizers
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, default="google/flan-t5-small")
    parser.add_argument("--task", type=str, default="all", help="multiple_choice, free_form or all")
    parser.add_argument("--data_dir", "-d", type=str, default="bbh")
    parser.add_argument("--prompts_dir", "-s", type=str, default="cot-prompts")
    parser.add_argument("--debug", type=bool, default=False)
    parser.add_argument('--num_beams', default=1, type=int)
    parser.add_argument('--seq_len', type=int, default=256, help='seq len')

    args = parser.parse_args()

    try:
        main(args)
    except torch.cuda.OutOfMemoryError:
        print(os.system("nvidia-smi"))
