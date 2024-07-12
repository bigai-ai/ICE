from copy import deepcopy
from typing import Any, Dict, List, Tuple
from collections import deque
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForCausalLM, AutoTokenizer

from ...util import nethook

from .ft_hparams import FTHyperParams
# from .visual_llm import show_logits


def apply_ft_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: FTHyperParams,
    copy=False,
    return_orig_weights=False,
    keep_original_weight=False,
    **kwargs: Any,
) -> Tuple[AutoModelForCausalLM, Dict[str, Any]]:
    """
    Returns a model with the desired changes.
    :param copy: If true, will preserve the original model while creating a new one to edit.
        Note that you are responsible for deallocating the new model's memory to avoid leaks.
    :return: (1) the updated model, (2) the weights that changed
    """
    weights_copy = {}
    if copy:
        model = deepcopy(model)

    deltas = execute_ft(model, tok, requests, hparams)

    with torch.no_grad():
        for w_name, upd_matrix in deltas.items():
            w = nethook.get_parameter(model, w_name)
            if return_orig_weights and w_name not in weights_copy:
                weights_copy[w_name] = w.detach().clone()

            w[...] += upd_matrix

    print(f"New weights successfully inserted into {list(deltas.keys())}")

    if not keep_original_weight:
        weights_copy = {}

    return model, weights_copy


def execute_ft(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: FTHyperParams,
    **kwargs: Any,
) -> Dict[str, Tuple[torch.Tensor]]:
    """
    Executes the FT update algorithm for the specified update at the specified layer
    Invariant: model at beginning of function == model at end of function
    """
    device = torch.device(f'cuda:{hparams.device}')
    # model = model.to(device)
    # Update target and print info
    requests = deepcopy(requests)
    for request in requests:
        if request["target_new"] != " ":
            # Space required for correct tokenization
            request["target_new"] = " " + request["target_new"]
        print(
            f"Executing FT algo for: "
            f"[{request['prompt']}] -> [{request['target_new']}]"
        )
    
    # Retrieve weights that user desires to change
    weights = {
        n: p
        for n, p in model.named_parameters()
        for layer in hparams.layers
        if hparams.rewrite_module_tmp.format(layer) in n
    }
    
    # Save old weights for future restoration
    weights_copy = {k: v.detach().clone() for k, v in weights.items()}
    print(f"Weights to be updated: {list(weights.keys())}")

    # Define inputs
    texts = [r["prompt"] for r in requests]
    targets = [r["target_new"] for r in requests]
    context_provided = 'context' in requests[0] and requests[0]['context'] is not None
    if context_provided:
        knowledge = [r["context"] for r in requests]
        context_num = len(knowledge[0])
    else:
        knowledge = [[r["prompt"] + r["target_new"] + '. '] for r in requests]
        context_num = 1
    
    # Configure optimizer / gradients
    opt = torch.optim.Adam(
        [v for _, v in weights.items()],
        lr=hparams.lr,
        weight_decay=hparams.weight_decay,
    )
    for name, w in model.named_parameters():
        w.requires_grad = name in weights

    # Update loop: intervene at layers simultaneously
    loss_meter = AverageMeter()
    for it in range(hparams.num_steps):
        print(20 * "=")
        print(f"Epoch: {it}")
        print(20 * "=")
        loss_meter.reset()

        for id, (txt, tgt, ctx) in enumerate(zip(
            chunks(texts, hparams.batch_size), chunks(
                targets, hparams.batch_size), chunks(knowledge, hparams.batch_size)
        )):
            inputs = tok(txt, return_tensors="pt", padding=True).to(device)
            target_ids = tok(tgt, return_tensors="pt", padding=True)["input_ids"].to(
                device
            )
            if hparams.objective_optimization == 'prompt_last':
                last_token_inds = inputs["attention_mask"].sum(dim=1) - 1
                loss_mask = target_ids != tok.unk_token_id
            elif hparams.objective_optimization == 'target_new':
                inputs_targets = [txt_ + tgt_ for txt_, tgt_ in zip(txt, tgt)]
                inputs_targets = tok(inputs_targets, return_tensors="pt", padding=True).to(device)
                num_prompt_toks = [int((i != tok.pad_token_id).sum()) for i in inputs['input_ids'].cpu()]
                num_pad_toks = [int((i == tok.pad_token_id).sum()) for i in inputs_targets['input_ids'].cpu()]
                prompt_len = [x + y for x, y in zip(num_pad_toks, num_prompt_toks)]
                prompt_target_len = inputs_targets['input_ids'].size(1)
                label_mask = torch.tensor([[False] * length + [True] * (prompt_target_len - length) for length in prompt_len]).to(device)
            elif hparams.objective_optimization in [ 'target_with_context','target_and_completion_with_context']:
                context_instuct = [
                    '',
                    'Fact: '
                    'A new knowledge: '
                    'Consider this new knowledge: ',
                    'Remember this fact: ',
                    'Remember this new fact and reason about its consequences: ',
                ]

                # Prepare tokens for targets, prompts+targets, and contexts+prompts+targets
                tgt = [t for t in tgt for _ in range(context_num)]
                txt = [t for t in txt for _ in range(context_num)]
                contexts = [random.sample(context_instuct, k=1)[0] + context + ' Now: '
                            for ctx_ in ctx
                            for context in ctx_]
                prompts_targets = [txt_ + tgt_ for txt_, tgt_ in zip(txt, tgt)]
                contexts_prompts_targets = [context + prompt_target for context, prompt_target in zip(contexts, prompts_targets)]
                contexts_prompts = [context + txt_ for context, txt_ in zip(contexts, txt) ]
                
                # Tokenize the above
                target_tokens = tok(tgt, return_tensors="pt", padding=True).to(device)
                prompts_targets = tok(prompts_targets, return_tensors="pt", padding=True).to(device)
                contexts_prompts_targets = tok(contexts_prompts_targets, return_tensors="pt", padding=True).to(device)
                contexts_prompts = tok(contexts_prompts, return_tensors="pt",padding=True).to(device)
                prompts = tok(txt, return_tensors="pt",padding=True).to(device)

                # Prepare label mask for targets
                prompt_target_len = prompts_targets['input_ids'].size(1)
                num_tgt_tokens = [int((i != tok.pad_token_id).sum()) for i in target_tokens['input_ids'].cpu()]
                label_mask = torch.tensor([[False] * (prompt_target_len - tgt_length)  + [True] * tgt_length for tgt_length in num_tgt_tokens]).to(device)
                if hparams.sample_with_context:
                    prefix = contexts_prompts_targets
                else:
                    prefix = prompts_targets

            else:
                print(f"{hparams.objective_optimization} has not been supported yet.")
                raise NotImplementedError
            # last_token_inds = inputs["attention_mask"].sum(dim=1) - 1
            # loss_mask = inputs != tok.unk_token_id
            # loss_mask = [:, ]
            opt.zero_grad()
            bs = inputs["input_ids"].shape[0]
            if 't5' in hparams.model_name.lower():
                inputs['decoder_input_ids'] = target_ids
                logits = model(**inputs).logits
                unmasked_log_probs = logits.log_softmax(-1).gather(-1, inputs['decoder_input_ids'].unsqueeze(-1)).squeeze(-1)

                mask = inputs['decoder_input_ids'] != -100
                n_tokens = mask.float().sum()
                avg_log_prob = (unmasked_log_probs * mask.float()).sum() / n_tokens
                nll = -avg_log_prob
                loss = nll
            elif 'chatglm' in hparams.model_name.lower():
                # def get_masks(seq, bos_token_id):
                #     """  code from model_chatglm.py  """
                #     if seq.count(bos_token_id) == 2:
                #         context_length = seq[2:].index(bos_token_id) + 2
                #     else:
                #         context_length = seq.index(bos_token_id)
                #     attention_mask = torch.ones((1, len(seq), len(seq)))
                #     attention_mask.tril_()
                #     attention_mask[..., :context_length] = 1
                #     # attention_mask.unsqueeze_(1)
                #     attention_mask = (attention_mask < 0.5).bool()
                #     return attention_mask

                input_ids = inputs['input_ids'].tolist()
                labels = target_ids.tolist()
                assert len(input_ids) == len(labels)
                len_batches = [len(input_ids[i]) + len(labels[i]) + 1
                                 for i in range(len(input_ids))]
                len_max_batch = max(len_batches)
                batch_input_ids = []
                batch_attention_mask = []
                batch_labels = []
                for x, y in zip(input_ids, labels):
                    len_padding = len_max_batch - len(x) - len(y)
                    if tok.padding_side and tok.padding_side == "left":
                        batch_label = [-100] * len_padding + [-100] * len(x) + y
                        batch_input_id = [0] * (len_padding) + x + y
                    else:
                        batch_label = [-100] * len(x) + y + [-100] * len_padding
                        batch_input_id = x + y + [0] * (len_padding)

                    # tensor_attention_mask = get_masks(batch_input_id, bos_token_id=64792)
                    tensor_input_ids = torch.tensor(batch_input_id, dtype=torch.long)
                    tensor_labels = torch.tensor(batch_label, dtype=torch.long)
                    batch_input_ids.append(tensor_input_ids)
                    # batch_attention_mask.append(tensor_attention_mask)
                    batch_labels.append(tensor_labels)
                # batch_attention_mask = torch.stack(batch_attention_mask).to(device)
                batch_input_ids = torch.stack(batch_input_ids).to(device)
                batch_labels = torch.stack(batch_labels).to(device)
                # loss = model(input_ids=batch_input_ids, labels=batch_labels).loss
                lm_logits = model(input_ids=batch_input_ids)['logits']
                lm_logits = lm_logits.to(torch.float32)
                shift_logits = lm_logits[..., :-1, :].contiguous()
                shift_labels = batch_labels[..., 1:].contiguous()
                # Flatten the tokens
                loss_fct = CrossEntropyLoss(ignore_index=-100)
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                loss = loss.to(lm_logits.dtype)
            else:
                if hparams.objective_optimization == 'prompt_last':
                    probs = torch.nn.functional.log_softmax(
                        model(**inputs).logits[torch.arange(bs), last_token_inds], dim=-1
                    )
                    loss = -(torch.gather(probs, 1, target_ids) * loss_mask).sum(
                        1
                    ) / loss_mask.sum(1)
                    loss = loss.mean()
                elif hparams.objective_optimization == 'target_new':
                    logits = model(**inputs_targets).logits
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = inputs_targets['input_ids'][..., 1:].contiguous()
                    loss_fct = CrossEntropyLoss(reduction='none')
                    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                    loss = loss.view(bs, -1)
                    loss = (loss * label_mask[:,1:]).sum(1) / label_mask[:,1:].sum(1)
                    loss = loss.mean()
                elif hparams.objective_optimization == 'target_with_context':
                    logits = model(**prompts_targets).logits
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = prompts_targets['input_ids'][..., 1:].contiguous()

                    # Do not back-propagate through the context
                    logits_with_context = model(**contexts_prompts_targets).logits.detach()
                    shift_logits_with_context = logits_with_context[..., -logits.size(1):-1, :].contiguous()

                    # Compute in-context loss
                    kl_div = F.kl_div(F.softmax(shift_logits, dim=-1).log(), F.softmax(shift_logits_with_context, dim=-1), reduction='none').sum(-1)

                    loss = kl_div
                    loss = (loss * label_mask[:,1:]).sum(1) / label_mask[:,1:].sum(1)
                    loss = loss.mean()
                elif hparams.objective_optimization == "target_and_completion_with_context":
                    if not hparams.static_target and it%hparams.target_update_interval == 0:
                        generated_sequences = model.generate(
                            **prefix, do_sample=True, max_new_tokens=hparams.max_new_tokens, min_new_tokens=hparams.max_new_tokens, num_return_sequences=hparams.num_return_sequences, temperature=hparams.temperature)
                    elif hparams.static_target and it%hparams.target_update_interval == 0:
                        if it == 0:
                            generated_sequences_list = [
                                model.generate(
                                    **prefix,
                                    do_sample=True,
                                    max_new_tokens=hparams.max_new_tokens,
                                    min_new_tokens=hparams.max_new_tokens,
                                    num_return_sequences=hparams.num_return_sequences,
                                    temperature=hparams.temperature,
                                    output_scores=True,
                                    return_dict_in_generate=True,

                                )
                                for _ in range(hparams.num_steps)
                            ]
                        generated_sequences= generated_sequences_list[it]['sequences']
                    # Construct the sequences without context, i.e., prompts + targets + completions
                    completions = generated_sequences[..., -hparams.max_new_tokens:]
                    sequences = torch.cat((torch.repeat_interleave(prompts_targets['input_ids'], hparams.num_return_sequences, dim=0), completions), dim=1)

                    # Mask the tokens before the first token of the target
                    target_mask = torch.zeros_like(sequences).bool() 
                    target_mask[..., :label_mask.size(1)] = torch.repeat_interleave(label_mask, hparams.num_return_sequences, dim=0)

                    # Mask the tokens after eos token for completions
                    sequence_indices = torch.arange(sequences.size(-1)).unsqueeze(0).expand_as(sequences).to(device)
                    eos_positions = sequences.size(-1) - (sequences != tok.eos_token_id).flip(-1).float().argmax(dim=-1)
                    completion_mask = (sequence_indices < eos_positions.unsqueeze(1)) & (sequence_indices >= prompts_targets['input_ids'].size(-1))
                    
                    loss_mask = (target_mask | completion_mask)

                    # Compute the log probabilities of the sequences, make sure to align the sequences with the logits:
                    # 1. Shift the sequences by dropping the first token.
                    shift_sequences = sequences[..., 1:].contiguous()
                    shift_loss_mask = loss_mask[..., 1:].contiguous()
                    shift_target_mask = target_mask[..., 1:].contiguous()
                    shift_completion_mask = completion_mask[..., 1:].contiguous()

                    # 2. Shift the probabilites by dropping the last token.
                    shift_logits = model(input_ids=sequences).logits[:, :-1, :]
                    shift_q = torch.softmax(shift_logits, dim=-1).contiguous()
                    # shift_q = torch.gather(shift_q, -1, shift_sequences.unsqueeze(-1)).squeeze(-1)
                    shift_log_q = torch.log(shift_q)
                        
                    if hparams.print_kl:
                        assert hparams.sample_with_context
                        if hparams.static_target:
                            # Need to prepend logits of the prompts and targets
                            shift_logits_with_context = torch.stack(generated_sequences_list[it]['scores'], dim=1).detach()
                            shift_p = torch.softmax(shift_logits_with_context, dim=-1).to(device).contiguous()
                            shift_log_p = torch.log(shift_p)
                        else:
                            shift_logits_with_context = model(input_ids=generated_sequences).logits[:, -sequences.size(-1):-1, :]
                            shift_p = torch.softmax(shift_logits_with_context, dim=-1).contiguous()
                            shift_log_p = torch.log(shift_p)

                        mle_loss = -torch.gather(shift_log_p + shift_log_q, -1, shift_sequences.unsqueeze(-1)).squeeze(-1)
                        kl_div = F.kl_div(shift_log_q, shift_p.detach(), reduction='none').sum(-1)

                        loss = mle_loss * shift_target_mask + kl_div * shift_completion_mask
                        loss = loss.sum(-1)
                    else:
                        mle_loss = -torch.gather(shift_log_q, -1, shift_sequences.unsqueeze(-1)).squeeze(-1)
                        loss = (shift_loss_mask * mle_loss).sum(-1) / shift_loss_mask.sum(-1)
                    loss = loss.mean()
                else:
                    raise NotImplementedError
            print(f"Batch loss {loss.item()}")
            loss_meter.update(loss.item(), n=bs)

            if loss.item() >= 1e-2 or hparams.print_kl:
                loss.backward()
                opt.step()

            if type(hparams.norm_constraint) is float:
                eps = hparams.norm_constraint
                with torch.no_grad():
                    for k, v in weights.items():
                        v[...] = torch.clamp(
                            v, min=weights_copy[k] - eps, max=weights_copy[k] + eps
                        )

        print(f"Total loss {loss_meter.avg}")

        if loss_meter.avg < 1e-2 and not hparams.print_kl:
            break

    deltas = {k: (weights[k] - weights_copy[k]).detach() for k in weights}

    # Restore state of original model
    with torch.no_grad():
        for k, v in weights.items():
            v[...] = weights_copy[k]

    print(f"Deltas successfully computed for {list(weights.keys())}")

    return deltas


def chunks(arr, n):
    """Yield successive n-sized chunks from arr."""
    chunk = []
    for a in arr:
        chunk.append(a)
        if len(chunk) == n:
            yield chunk
            chunk = []
    if len(chunk) > 0:
        yield chunk


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
