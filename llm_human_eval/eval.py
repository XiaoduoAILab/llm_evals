import logging
import os
import random
import re
import warnings

import numpy as np
import torch
import transformers
from tqdm import tqdm
from transformers import StoppingCriteriaList

from human_eval.data import HUMAN_EVAL
from human_eval.data import write_jsonl, read_problems
from human_eval.evaluation import evaluate_functional_correctness


def clean_code(text):
    if '\n\n\n' in text:
        text = text.split('\n\n\n')[0]
    else:
        pattern = '[\n][\n][\S]'
        result = re.search(pattern, text)
        if result:
            position = result.span()[0]
            text = text[:position]

    return text


def custom_stopping_criteria(input_ids: torch.LongTensor, score: torch.FloatTensor, **kwargs) -> bool:
    # last_token = input_ids[0][-1]
    # for stop in stop_words_ids:
    #     if tokenizer.decode(stop) == tokenizer.decode(last_token):
    #         return True
    if input_ids[0][-1] == 13 and input_ids[0][-2] == 13 and input_ids[0][-3] == 13:
        return True

    return False


@torch.no_grad()
def generate_one_completion(prompt):
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(inputs, max_new_tokens=args.seq_len, pad_token_id=tokenizer.eos_token_id,
                             stopping_criteria=StoppingCriteriaList([custom_stopping_criteria]))
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    text = text[len(prompt):]
    if args.debug:
        print('======================================')
        print('prompt:' + str(prompt))
        print('text:\n' + str(text))

    text = clean_code(text)
    if args.debug:
        print('text (after clean):\n' + str(text))

    return text


def get_logger():
    logname = os.path.join(args.model, 'llm_human_eval.txt')
    if os.path.exists(logname):
        os.remove(logname)

    logging.basicConfig(filename=logname,
                        filemode='a',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.DEBUG)

    logger = logging.getLogger()
    return logger


def main(args):
    problems = read_problems()
    sample_file = os.path.join(args.model, 'llm_human_eval.txt')

    # num_samples_per_task = 200
    num_samples_per_task = 1
    samples = []

    for task_id in tqdm(problems):
        for _ in range(num_samples_per_task):
            completion = generate_one_completion(problems[task_id]["prompt"])
            samples.append(dict(task_id=task_id, completion=completion))

    write_jsonl(sample_file, samples)

    k = "1,10,100"
    n_workers = 1
    timeout = 3.0
    problem_file = HUMAN_EVAL

    """
    Evaluates the functional correctness of generated samples, and writes
    results to f"{sample_file}_results.jsonl.gz"
    """
    k = list(map(int, k.split(",")))
    results = evaluate_functional_correctness(sample_file, k, n_workers, timeout, problem_file)
    print(results)
    log(results)


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

    import argparse

    parser = argparse.ArgumentParser(description='Human Eval')
    # general
    parser.add_argument('--model', type=str, default='/mnt/sdb/ly/models/hf_converted_llama/7B/', help='')
    parser.add_argument('--seq_len', type=int, default=320, help='seq len')
    parser.add_argument('--num_beams', default=1, type=int)
    parser.add_argument('--debug', type=bool, default=False)

    args = parser.parse_args()

    model_config = transformers.AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    print(model_config)

    torch_dtype = model_config.torch_dtype
    if torch_dtype == torch.float32 or torch_dtype == torch.float64:
        torch_dtype = torch.float16

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model, skip_special_tokens=True, add_bos_token=False,
                                                           add_eos_token=False, clean_up_tokenization_spaces=True,
                                                           use_fast=False, trust_remote_code=True)

    print(f"Loading model from {args.model}")
    model = transformers.AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch_dtype, device_map='auto',
                                                              trust_remote_code=True)
    model.eval()

    print(model)
    print(tokenizer)

    logger = get_logger()


    def log(msg):
        logger.info(msg)


    # Generate
    try:
        main(args)
    except torch.cuda.OutOfMemoryError:
        print(os.system("nvidia-smi"))
