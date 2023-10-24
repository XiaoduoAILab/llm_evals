import json
import logging
import os
import random
import warnings

import numpy as np
import torch
import transformers
from tqdm import trange
from transformers import StoppingCriteriaList

from dataset import get_examples, extract_answer, INVALID_ANS


def custom_stopping_criteria(input_ids: torch.LongTensor, score: torch.FloatTensor, **kwargs) -> bool:
    # last_token = input_ids[0][-1]
    # for stop in stop_words_ids:
    #     if tokenizer.decode(stop) == tokenizer.decode(last_token):
    #         return True
    if input_ids[0][-1] == 13 and input_ids[0][-2] == 13:
        return True

    return False


@torch.no_grad()
def generate(prompt):
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    num_tokens = inputs.shape[-1]
    outputs = model.generate(inputs, max_new_tokens=args.seq_len, pad_token_id=tokenizer.eos_token_id,
                             stopping_criteria=StoppingCriteriaList([custom_stopping_criteria]))
    text = tokenizer.decode(outputs[0][num_tokens:]).strip()
    return text


def get_logger():
    model_name = os.path.basename(os.path.dirname(args.model))
    logname = '{}_gsm8k.jsonl'.format(model_name)
    if os.path.exists(logname):
        os.remove(logname)

    logging.basicConfig(filename=logname,
                        filemode='a',
                        format='%(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.DEBUG)

    logger = logging.getLogger()
    return logger


def main(args):
    splitter = '\n\n'
    num_correct = 0
    num_processed = 0

    test_examples = get_examples("test", args.use_cn)
    num_examples = len(test_examples)
    prompt = open(args.prompt_file, 'r').read()

    t = trange(num_examples, desc='', leave=True)
    for i in t:
        example = test_examples[i]
        qn = example["question"].strip()
        an = example["answer"].strip()
        input_text = '{}\nQuestion: {}\nLet\'s think step by step\n'.format(prompt, qn)
        # print("==========================================")
        # print(input_text)

        output_text = generate(input_text)
        if splitter in output_text:
            output_text = output_text.split(splitter)[0]
        model_output = extract_answer(output_text.replace('$', ''))
        gt_answer = extract_answer(example["answer"])

        # print('Answer: {}'.format(gt_answer))
        # print('Model output: {}'.format(model_output))

        num_processed += 1

        correct = False
        if model_output != INVALID_ANS:
            try:
                correct = eval(model_output) == eval(gt_answer)
            except Exception as ex:
                print(ex)

        if correct:
            num_correct += 1

        bar_desc = 'accuracy: {:.4f}%'.format(num_correct * 100 / num_processed)
        t.set_description(bar_desc, refresh=True)

        log_line = json.dumps(
            dict(question=qn, answer=an, output_text=output_text, model_output=model_output, gt_answer=gt_answer,
                 correct=correct))
        log(log_line)

    result = 'GSM8k test accuracy: %.2f' % (num_correct * 100 / num_examples)

    log(result)
    print(result)


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

    parser = argparse.ArgumentParser(description='GSM8k')
    # general
    parser.add_argument('--model', type=str, default='/mnt/sdb/ly/models/hf_converted_llama/7B/', help='')
    parser.add_argument('--prompt_file', type=str, default='lib_prompt/prompt_original_5_shot.txt')
    parser.add_argument('--seq_len', type=int, default=256, help='seq len')
    parser.add_argument('--num_beams', default=1, type=int)
    parser.add_argument('--do_sample', type=bool, default=False, help='do sample')
    parser.add_argument('--use_cn', type=bool, default=False, help='Chinese version')
    parser.add_argument('--compile', type=bool, default=False, help='compile')
    parser.add_argument('--rope_scaling', type=str, default='None')

    args = parser.parse_args()
    print(args)

    model_config = transformers.AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    print(model_config)

    torch_dtype = model_config.torch_dtype
    if torch_dtype == torch.float32 or torch_dtype == torch.float64:
        torch_dtype = torch.float16

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model,
                                                           skip_special_tokens=True,
                                                           add_bos_token=False,
                                                           add_eos_token=False,
                                                           clean_up_tokenization_spaces=True,
                                                           use_fast=False,
                                                           trust_remote_code=True)

    kwargs = {}
    if args.rope_scaling != "None":
        kwargs['rope_scaling'] = eval(args.rope_scaling)

    print(f"Loading model from {args.model}")
    model = transformers.AutoModelForCausalLM.from_pretrained(args.model,
                                                              torch_dtype=torch_dtype,
                                                              device_map='auto',
                                                              trust_remote_code=True,
                                                              **kwargs)

    if args.compile:
        model = torch.compile(model)

    model.eval()

    print(model)
    print(tokenizer)

    logger = get_logger()


    def log(msg):
        logger.info(msg)


    ### Generate
    try:
        main(args)
    except torch.cuda.OutOfMemoryError:
        print(os.system("nvidia-smi"))
