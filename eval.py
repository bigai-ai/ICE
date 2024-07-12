import os
import torch
import torch.nn.functional as F
import typing
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import collections
import argparse


def parse_args(args=None):
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name_or_path", default="", type=str, help="Path to pre-trained model")
    parser.add_argument("--output_file", default="", type=str, help="Generated sentences file")
    parser.add_argument("--result_file", default="", type=str, help="Result file")
    return parser.parse_args(args)

def PPL_new_target(
        model,
        tok,
        prompt: typing.Union[str, typing.List[str]],
        target_new: typing.Union[str, typing.List[str]],
        device,
):
    sampled_token_ids = tok(prompt[0] + target_new, return_tensors="pt", padding=True, truncation=True)["input_ids"].to(
        device)
    with torch.no_grad():
        outputs = model(input_ids=sampled_token_ids)
        logits = outputs['logits']
        log_probs = torch.gather(F.softmax(logits, dim=-1).log2(), -1, sampled_token_ids.unsqueeze(-1)).squeeze(-1)
    return torch.exp2(-log_probs.mean())


def PPL_new(
        model,
        tok,
        prompt: typing.Union[str, typing.List[str]],
        target_new: typing.Union[str, typing.List[str]],
        generation_sentence: typing.Union[str, typing.List[str]],
        device,
):
    prompt_ids = tok(prompt, return_tensors="pt", padding=True, truncation=True)["input_ids"].to(device)
    target_ids = tok(target_new, return_tensors="pt", padding=True, truncation=True)["input_ids"].to(device)
    generation_sentence_ids = tok(generation_sentence, return_tensors="pt", padding=True, truncation=True)[
        "input_ids"].to(device)
    target_mask = torch.ones_like(generation_sentence_ids).bool()
    target_mask[..., :prompt_ids.size(-1)] = False
    with torch.no_grad():
        outputs = model(input_ids=generation_sentence_ids)
        logits = outputs['logits']
        log_probs = torch.gather(F.softmax(logits, dim=-1).log2(), -1, generation_sentence_ids.unsqueeze(-1)).squeeze(
            -1)

    log_probs = torch.masked_select(log_probs, target_mask)
    ppl = torch.exp2(-log_probs.mean())
    return ppl


def eval(result_path):
    if os.path.exists(result_path):

        with open(result_path, 'r') as file:
            datas = json.load(file)

        Edit_Succ_list = [data_rome_counterfact['post']['rewrite_acc'][0] for data_rome_counterfact in datas]
        Edit_Succ = sum(Edit_Succ_list) / len(Edit_Succ_list) * 100
        print('Edit_Succ:', Edit_Succ)

        Portability_list = []
        portability_dict = collections.defaultdict(list)
        for data_rome_counterfact in datas:
            metrics = []
            for key in data_rome_counterfact['post']['portability'].keys():
                metrics = metrics + data_rome_counterfact['post']['portability'][key]
                portability_dict[key].extend(data_rome_counterfact['post']['portability'][key])
            if len(metrics) == 0:
                continue
            portability = sum(metrics) / len(metrics) * 100
            Portability_list.append(portability)
        if len(Portability_list) == 0:
            print('Portability:', 0)
        else:
            Portability = sum(Portability_list) / len(Portability_list)
            print('Portability:', Portability)
        for key in portability_dict.keys():
            portability = sum(portability_dict[key]) / len(portability_dict[key]) * 100
            print(f'Portability ({key}):', portability)

        Locality_list = []
        for data_rome_counterfact in datas:
            metrics = []
            for key in data_rome_counterfact['post']['locality'].keys():
                metrics = metrics + data_rome_counterfact['post']['locality'][key]
            if len(metrics) == 0:
                continue
            locality = sum(metrics) / len(metrics) * 100
            Locality_list.append(locality)
        if len(Locality_list) == 0:
            print('Locality:', 0)
        else:
            Locality = sum(Locality_list) / len(Locality_list)
            print('Locality:', Locality)

        Fluency_list = [x['post']['fluency']['ngram_entropy'] for x in datas]
        Fluency = sum(Fluency_list) / len(Fluency_list) * 100
        print('Fluency:', Fluency)


if __name__ == '__main__':
    args = parse_args()
    model_path = args.model_name_or_path

    device = 'cuda:0'
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    tok = AutoTokenizer.from_pretrained(model_path)
    tok.pad_token_id = tok.eos_token_id

    with open(args.result_file, 'r') as f:
        result = json.load(f)
    with open(args.output_file, 'r') as f:
        generated_sentence = json.load(f)
    sum_ppl = 0
    ds_size = 0
    for id in range(len(result)):
        prompt = result[id]['requested_rewrite']['prompt']
        target_new = result[id]['requested_rewrite']['target_new']
        generation_sentence = generated_sentence[id]['generation_sentence']
        ppl_new_target = PPL_new_target(model, tok, prompt, target_new, device)
        ppl_gen_sentence = PPL_new(model, tok, prompt, target_new, generation_sentence, device)
        normalized_ppl = ppl_gen_sentence / ppl_new_target
        if torch.isnan(normalized_ppl):
            continue
        if normalized_ppl.item() > 500:
            print(normalized_ppl)
        sum_ppl += normalized_ppl
        ds_size += 1
    print(ds_size)
    print(eval(args.result_file))
    print('ppl_r: ', sum_ppl / ds_size)
