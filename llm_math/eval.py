import json
import os
import random
import warnings

import numpy as np
import torch
import transformers
from tqdm import trange

from dataset.util import last_boxed_only_string
from math_equivalence import is_equiv


@torch.no_grad()
def generate_one_completion(prompt, model, tokenizer, num_tokens=20):
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(inputs, max_new_tokens=num_tokens, pad_token_id=tokenizer.eos_token_id)
    num_tokens = inputs.shape[-1]
    text = tokenizer.decode(outputs[0][num_tokens:]).strip()

    return text


def call_engine(train_prompt, problem, model, tokenizer):
    '''
    Given a problem, returns the most likely answer determined by the engine
    '''
    test_question = "\n###\nProblem: " + problem + "\n" + "Answer: $"
    prompt = train_prompt + test_question

    if args.debug:
        print('prompt: ' + str(prompt))
        print('len(prompt): ' + str(len(prompt)))

    output_str = generate_one_completion(prompt, model, tokenizer)
    output_full = output_str

    startindex = 0
    endindex = 1
    for token in output_full[startindex + 1:]:
        if token == "$" or token == "#" or token == "\n":
            break
        else:
            endindex += 1

    output = output_full[startindex:endindex]
    return output_full, output


def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[:len(left)] == left
        assert s[-1] == "}"
        return s[len(left):-1]
    except:
        return None


def run(model, tokenizer, max=-1):
    outputs = []
    answers = []
    types = []
    levels = []

    train_prompt = open(args.prompt_file, 'r').read()

    fnames_list = []
    finfo_list = []

    cors = {}
    subject_cors = {}
    level_cors = {}
    correct = 0
    total = 0
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            fnames_list.append(os.path.join(subdir, file))
            finfo_list.append((subdir, file))

    num_examples = len(finfo_list)
    print('num_examples: ' + str(num_examples))

    t = trange(num_examples, desc='', leave=True)
    for i in t:
        subdir, file = finfo_list[i]
        with open(os.path.join(subdir, file), 'r') as fp:
            try:
                problem_data = json.load(fp)
            except Exception as e:
                print(f"Error loading JSON from {file}", e)
                raise e
            prob_level = problem_data["level"]
            prob_type = problem_data["type"]
            try:
                prob_level = int(prob_level.split("Level ")[1])
            except:
                prob_level = None
            output_full, output = call_engine(train_prompt, problem_data["problem"], model, tokenizer)
            answer = remove_boxed(last_boxed_only_string(problem_data["solution"]))

            levels.append(prob_level)
            types.append(prob_type)
            outputs.append(output)
            answers.append(answer)

            if args.debug:
                print("Model output:")
                # print(output_full)
                print(output)
                print("Correct answer:")
                print(answer)
                print("--------------------------------------------")

            try:
                equiv = is_equiv(output, answer)
            except:
                equiv = False
            if (prob_level, prob_type) in cors:
                cors[(prob_level, prob_type)].append(equiv)
            else:
                cors[(prob_level, prob_type)] = [equiv]
            if prob_level in level_cors:
                level_cors[prob_level].append(equiv)
            else:
                if prob_level is not None:
                    level_cors[prob_level] = [equiv]
            if prob_type in subject_cors:
                subject_cors[prob_type].append(equiv)
            else:
                if prob_type is not None:
                    subject_cors[prob_type] = [equiv]
            if equiv:
                correct += 1
            total += 1

            # print(str(correct) + "/" + str(total))

            bar_desc = 'accuracy: {:.4f}%'.format(correct * 100 / total)
            t.set_description(bar_desc, refresh=True)

    with open("{}/outputs_answers.txt".format(args.model), "w+") as f:
        for k, (output, answer, prob_type, prob_level, fname) in enumerate(
                zip(outputs, answers, types, levels, fnames_list)):
            f.write("{} TYPE: {} | LEVEL: {} | OUTPUT: {} | ANSWER: {} | FNAME: {}\n".format(k, prob_type, prob_level,
                                                                                             output, answer, fname))

        f.write("#####################\n")
        # also get accuracies for each
        for subject in ['Prealgebra', 'Algebra', 'Number Theory', 'Counting & Probability', 'Geometry',
                        'Intermediate Algebra', 'Precalculus']:
            for level in range(1, 6):
                key = (level, subject)
                if key not in cors.keys():
                    print("Skipping", key)
                    continue
                cors_list = cors[key]
                print("{} Level {} Accuracy = {}/{} = {:.3f}".format(subject, level, np.sum(cors_list), len(cors_list),
                                                                     np.mean(cors_list)))
                f.write(
                    "{} Level {} Accuracy = {}/{} = {:.3f}\n".format(subject, level, np.sum(cors_list), len(cors_list),
                                                                     np.mean(cors_list)))
        print("#####################")
        f.write("#####################\n")
        for level in sorted(level_cors):
            if level not in level_cors.keys():
                print("Skipping", level)
                continue
            cors_list = level_cors[level]
            print("Level {} Accuracy = {}/{} = {:.3f}".format(level, np.sum(cors_list), len(cors_list),
                                                              np.mean(cors_list)))
            f.write("Level {} Accuracy = {}/{} = {:.3f}\n".format(level, np.sum(cors_list), len(cors_list),
                                                                  np.mean(cors_list)))
        print("#####################")
        f.write("#####################\n")
        for subject in ['Prealgebra', 'Algebra', 'Number Theory', 'Counting & Probability', 'Geometry',
                        'Intermediate Algebra', 'Precalculus']:
            if subject not in subject_cors.keys():
                print("Skipping", subject)
                continue
            cors_list = subject_cors[subject]
            print("{} Accuracy = {}/{} = {:.3f}".format(subject, np.sum(cors_list), len(cors_list), np.mean(cors_list)))
            f.write(
                "{} Accuracy = {}/{} = {:.3f}\n".format(subject, np.sum(cors_list), len(cors_list), np.mean(cors_list)))
        print("#####################")
        f.write("#####################\n")
        print("Overall Accuracy = {}/{} = {:.3f}".format(correct, total, correct / total))
        f.write("Overall Accuracy = {}/{} = {:.3f}\n".format(correct, total, correct / total))


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

    parser = argparse.ArgumentParser(description='MATH dataset')
    # general
    parser.add_argument('--model', type=str, default='/mnt/sdb/ly/models/hf_converted_llama/7B/', help='')
    parser.add_argument('--num_beams', default=1, type=int)
    parser.add_argument('--prompt_file', type=str, default='lib_prompt/prompt_ao_5_shot.txt')
    parser.add_argument('--debug', default=False, type=bool)
    parser.add_argument('--b16', type=bool, default=False, help='b16')

    # Dataloading
    parser.add_argument('--math_dataroot', default='./MATH/test/*/*.json', type=str)

    # Others
    parser.add_argument('--workers', default=4, type=int)

    args = parser.parse_args()
    args.load = args.model

    rootdir = "./MATH/test"

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model, skip_special_tokens=False, add_bos_token=False,
                                                           add_eos_token=False, clean_up_tokenization_spaces=True,
                                                           use_fast=False,
                                                           trust_remote_code=True)
    print(f"Loading model from {args.model}")
    model_config = transformers.AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    print(model_config)

    torch_dtype = model_config.torch_dtype
    if torch_dtype == torch.float32 or torch_dtype == torch.float64:
        torch_dtype = torch.float16

    model = transformers.AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch_dtype, device_map='auto',
                                                              trust_remote_code=True)
    print(f"Successfully loaded model from {args.model}")

    if torch.__version__.startswith('2.0'):
        model = torch.compile(model)

    model.eval()

    print(model)
    print(tokenizer)

    ### Generate
    try:
        run(model, tokenizer)
    except torch.cuda.OutOfMemoryError:
        print(os.system("nvidia-smi"))
