import argparse
import os
import random
import warnings

import numpy as np
import pandas as pd
import torch
import transformers
from tqdm import tqdm

from categories import subcategories, categories

choices = ["A", "B", "C", "D"]


def format_subject(subject):
    return subject.replace('_', ' ')


def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j + 1])
    prompt += "\nAnswer:  "
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    return prompt


def gen_prompt(train_df, subject, k=-1):
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(
        format_subject(subject)
    )
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt


@torch.no_grad()
def eval(args, subject, model, tokenizer, dev_df, test_df):
    cors = []
    all_probs = []
    all_labels = []
    answers = choices[: test_df.shape[1] - 2]

    for i in range(test_df.shape[0]):
        # get prompt and make sure it fits
        k = args.ntrain
        prompt_end = format_example(test_df, i, include_answer=False)
        train_prompt = gen_prompt(dev_df, subject, k)
        prompt = train_prompt + prompt_end

        inputs = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
        # print('inputs: ' + str(inputs))
        # print('inputs.size(): ' + str(inputs.size()))
        # print('inputs.shape: ' + str(inputs.shape))

        while inputs.shape[-1] > args.context_size:
            k -= 1
            train_prompt = gen_prompt(dev_df, subject, k)
            prompt = train_prompt + prompt_end
            inputs = tokenizer.encode(prompt, return_tensors="pt").to(model.device)

        # print('=================================================')
        # print('prompt: ' + str(prompt))

        label = test_df.iloc[i, test_df.shape[1] - 1]

        # decoder_input_ids = tokenizer("", return_tensors="pt").input_ids.to(model.device)
        # decoder_input_ids = model._shift_right(decoder_input_ids)
        outputs = model.generate(inputs, max_new_tokens=1, return_dict_in_generate=True, output_scores=True,
                                 pad_token_id=tokenizer.eos_token_id)

        # print('outputs: ' + str(outputs))
        # print('outputs.scores: ' + str(outputs.scores))
        # print('outputs.scores[0]: ' + str(outputs.scores[0]))
        logits = outputs.scores[0][0]
        # print('torch.sum(logits): ' + str(torch.sum(logits)))
        # print('logits[0]: ' + str(logits[0]))
        # print('logits[1]: ' + str(logits[1]))
        # print('logits[2]: ' + str(logits[2]))
        # print('logits.shape: ' + str(logits.shape))
        # print('logits: ' + str(logits))
        # print('logits.size(): ' + str(logits.size()))

        # sequence = outputs.sequences[0]
        # print('sequence: ' + str(sequence))
        # print('sequence.size(): ' + str(sequence.size()))
        # sequence = sequence.detach().cpu().numpy().tolist()
        # print('sequence: ' + str(sequence))
        # text = tokenizer.decode(sequence)
        # print('text: ' + str(text))

        # print('tokenizer("A"): ' + str(tokenizer("A")))
        # print('tokenizer("B"): ' + str(tokenizer("B")))
        # print('tokenizer("C"): ' + str(tokenizer("C")))
        # print('tokenizer("D"): ' + str(tokenizer("D")))
        #
        # print('tokenizer("A").input_ids: ' + str(tokenizer("A").input_ids))
        # print('tokenizer("B").input_ids: ' + str(tokenizer("B").input_ids))
        # print('tokenizer("C").input_ids: ' + str(tokenizer("C").input_ids))
        # print('tokenizer("D").input_ids: ' + str(tokenizer("D").input_ids))

        # print('tokenizer("A").input_ids[0]: ' + str(tokenizer("A").input_ids[0]))
        # print('tokenizer("B").input_ids[0]: ' + str(tokenizer("B").input_ids[0]))
        # print('tokenizer("C").input_ids[0]: ' + str(tokenizer("C").input_ids[0]))
        # print('tokenizer("D").input_ids[0]: ' + str(tokenizer("D").input_ids[0]))
        #
        # print('logits[tokenizer("A").input_ids[0]]: ' + str(logits[tokenizer("A").input_ids[0]]))
        # print('logits[tokenizer("B").input_ids[0]]: ' + str(logits[tokenizer("B").input_ids[0]]))
        # print('logits[tokenizer("C").input_ids[0]]: ' + str(logits[tokenizer("C").input_ids[0]]))
        # print('logits[tokenizer("D").input_ids[0]]: ' + str(logits[tokenizer("D").input_ids[0]]))

        logits = torch.nn.functional.softmax(logits, dim=0, dtype=torch.float32)
        probs = torch.tensor(
            [
                logits[tokenizer("A").input_ids[0]],
                logits[tokenizer("B").input_ids[0]],
                logits[tokenizer("C").input_ids[0]],
                logits[tokenizer("D").input_ids[0]],
            ]
        )
        # print('probs: ' + str(probs))
        # print('np.argmax(probs): ' + str(np.argmax(probs)))

        # probs = torch.nn.functional.softmax(probs, dim=0, dtype=torch.float32).detach().cpu().numpy()
        probs = probs.detach().cpu().numpy()
        # print('probs: ' + str(probs))
        # print('np.argmax(probs): ' + str(np.argmax(probs)))

        probs = (probs)
        # print('probs: ' + str(probs))
        # print('np.argmax(probs): ' + str(np.argmax(probs)))
        pred = {0: "A", 1: "B", 2: "C", 3: "D"}[np.argmax(probs)]
        # print('pred: ' + pred)

        cor = pred == label
        cors.append(cor)
        all_probs.append(probs)
        all_labels.append(label)

    acc = np.mean(cors)
    cors = np.array(cors)

    all_probs = np.array(all_probs)
    print("Average accuracy {:.3f} - {}".format(acc, subject))

    return cors, acc, all_probs, all_labels


def main(args):
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model, skip_special_tokens=True, add_bos_token=False,
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

    model.eval()

    print(model)
    print(tokenizer)

    # print('tokenizer.encode("A", skip_special_tokens=True): ' + str(tokenizer.encode("A", skip_special_tokens=True)))
    # print('tokenizer.encode("B", skip_special_tokens=True): ' + str(tokenizer.encode("B", skip_special_tokens=True)))
    # print('tokenizer.encode("C", skip_special_tokens=True): ' + str(tokenizer.encode("C", skip_special_tokens=True)))
    # print('tokenizer.encode("D", skip_special_tokens=True): ' + str(tokenizer.encode("D", skip_special_tokens=True)))

    subjects = sorted(
        [
            f.split("_test.csv")[0]
            for f in os.listdir(os.path.join(args.data_dir, "test"))
            if "_test.csv" in f
        ]
    )

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    if not os.path.exists(os.path.join(args.save_dir, "results_{}".format(args.model))):
        os.makedirs(os.path.join(args.save_dir, "results_{}".format(args.model)))

    all_cors = []
    subcat_cors = {
        subcat: [] for subcat_lists in subcategories.values() for subcat in subcat_lists
    }
    cat_cors = {cat: [] for cat in categories}
    all_probs = []
    all_labels = []

    for subject in tqdm(subjects):
        dev_df = pd.read_csv(
            os.path.join(args.data_dir, "dev", subject + "_dev.csv"), header=None
        )[: args.ntrain]
        test_df = pd.read_csv(
            os.path.join(args.data_dir, "test", subject + "_test.csv"), header=None
        )

        cors, acc, probs, labels = eval(args, subject, model, tokenizer, dev_df, test_df)
        all_probs.append(probs.tolist())
        all_labels.append(labels)
        subcats = subcategories[subject]
        for subcat in subcats:
            subcat_cors[subcat].append(cors)
            for key in categories.keys():
                if subcat in categories[key]:
                    cat_cors[key].append(cors)
        all_cors.append(cors)

        test_df["{}_correct".format(args.model)] = cors
        for j in range(probs.shape[1]):
            choice = choices[j]
            test_df["{}_choice{}_probs".format(args.model, choice)] = probs[:, j]
        test_df.to_csv(
            os.path.join(
                args.save_dir, "results_{}".format(args.model), "{}.csv".format(subject)
            ),
            index=None,
        )

    for subcat in subcat_cors:
        subcat_acc = np.mean(np.concatenate(subcat_cors[subcat]))
        print("Average accuracy {:.3f} - {}".format(subcat_acc, subcat))

    for cat in cat_cors:
        cat_acc = np.mean(np.concatenate(cat_cors[cat]))
        print("Average accuracy {:.3f} - {}".format(cat_acc, cat))
    weighted_acc = np.mean(np.concatenate(all_cors))
    print("Average accuracy: {:.3f}".format(weighted_acc))

    data = {'all_probs': all_probs, 'all_labels': all_labels}
    import pickle
    model_name = os.path.basename(os.path.dirname(args.model))
    filename = '{}_dumps.pkl'.format(model_name)
    with open(filename, 'wb') as fp:
        pickle.dump(data, fp)


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
    torch.manual_seed(42)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # To avoid warnings about parallelism in tokenizers
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", "-k", type=int, default=5)
    parser.add_argument("--ngpu", "-g", type=int, default=2)
    parser.add_argument("--data_dir", "-d", type=str, default="data")
    parser.add_argument("--save_dir", "-s", type=str, default="results")
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="google/flan-t5-small",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--context_size", type=int, default=2048)

    args = parser.parse_args()
    print(args)

    main(args)
