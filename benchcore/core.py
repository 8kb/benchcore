"""
CORE metric evaluation: functions for evaluating a model on the ICL benchmark tasks described in
the DCLM paper (https://arxiv.org/abs/2406.11794). Ported from nanochat/core_eval.py.

No ambient distributed state -- evaluate_task takes rank/world_size as explicit parameters
(mirroring datacore.DataManager.batches()) rather than reading torch.distributed directly; a
caller running under torchrun passes its own rank/world_size and calls dist.all_reduce itself, or
uses BenchManager.core() which does this for it (see manager.py).
"""
import random

import torch

from benchcore.prompts import (
    batch_sequences_lm,
    batch_sequences_mc,
    batch_sequences_schema,
    render_prompts_lm,
    render_prompts_mc,
    render_prompts_schema,
    stack_sequences,
)


@torch.no_grad()
def forward_model(model, input_ids):
    """
    Take BxT tensor of token ids, return BxT tensor of losses and argmax predictions.
    The last column of losses is set to nan because we don't have autoregressive targets there.
    """
    batch_size, seq_len = input_ids.size()
    outputs = model(input_ids)
    target_ids = torch.roll(input_ids, shifts=-1, dims=1)
    losses = torch.nn.functional.cross_entropy(
        outputs.view(batch_size * seq_len, -1),
        target_ids.view(batch_size * seq_len),
        reduction='none',
    ).view(batch_size, seq_len)
    losses[:, -1] = float('nan')
    predictions = outputs.argmax(dim=-1)
    return losses, predictions


@torch.no_grad()
def evaluate_example(idx, model, tokenizer, data, device, task_meta):
    """Evaluate a single example, return True if correct, False otherwise."""
    item = data[idx]
    task_type = task_meta['task_type']
    num_fewshot = task_meta['num_fewshot']
    continuation_delimiter = task_meta['continuation_delimiter']

    fewshot_examples = []
    if num_fewshot > 0:
        rng = random.Random(1234 + idx)
        available_indices = [i for i in range(len(data)) if i != idx]
        fewshot_indices = rng.sample(available_indices, num_fewshot)
        fewshot_examples = [data[i] for i in fewshot_indices]

    if task_type == 'multiple_choice':
        prompts = render_prompts_mc(item, continuation_delimiter, fewshot_examples)
        tokens, start_idxs, end_idxs = batch_sequences_mc(tokenizer, prompts)
    elif task_type == 'schema':
        prompts = render_prompts_schema(item, continuation_delimiter, fewshot_examples)
        tokens, start_idxs, end_idxs = batch_sequences_schema(tokenizer, prompts)
    elif task_type == 'language_modeling':
        prompts = render_prompts_lm(item, continuation_delimiter, fewshot_examples)
        tokens, start_idxs, end_idxs = batch_sequences_lm(tokenizer, prompts)
    else:
        raise ValueError(f"Unsupported task type: {task_type}")

    # Some models can't forward sequences beyond a certain length (e.g. GPT-2) -- max_seq_len is
    # an OPTIONAL Model attribute (see protocols.py); truncate and shift indices when present.
    max_seq_len = getattr(model, 'max_seq_len', None)
    if max_seq_len is not None:
        new_tokens, new_start_idxs, new_end_idxs = [], [], []
        for t, s, e in zip(tokens, start_idxs, end_idxs):
            if len(t) > max_seq_len:
                num_to_crop = len(t) - max_seq_len
                new_tokens.append(t[-max_seq_len:])
                new_start_idxs.append(s - num_to_crop)
                new_end_idxs.append(e - num_to_crop)
                assert s - num_to_crop >= 0, "this should never happen right?"
                assert e - num_to_crop >= 0, "this should never happen right?"
            else:
                new_tokens.append(t)
                new_start_idxs.append(s)
                new_end_idxs.append(e)
        tokens, start_idxs, end_idxs = new_tokens, new_start_idxs, new_end_idxs

    pad_token_id = tokenizer.get_bos_token_id() # use BOS as pad token is ok
    input_ids = stack_sequences(tokens, pad_token_id)
    input_ids = input_ids.to(device)

    losses, predictions = forward_model(model, input_ids)

    if task_type == 'language_modeling':
        si = start_idxs[0]
        ei = end_idxs[0]
        predicted_tokens = predictions[0, si - 1:ei - 1]
        actual_tokens = input_ids[0, si:ei]
        is_correct = torch.all(predicted_tokens == actual_tokens).item()
    elif task_type in ('multiple_choice', 'schema'):
        mean_losses = [losses[i, si - 1:ei - 1].mean().item()
                       for i, (si, ei) in enumerate(zip(start_idxs, end_idxs))]
        pred_idx = mean_losses.index(min(mean_losses))
        is_correct = pred_idx == item['gold']
    else:
        raise ValueError(f"Unsupported task type: {task_type}")

    return is_correct


def evaluate_task(model, tokenizer, data, device, task_meta, *, rank=0, world_size=1):
    """
    Evaluate one task across many examples, striding examples across ranks. Caller is responsible
    for the actual all_reduce when world_size > 1 (see BenchManager.core, which does it) --
    this function returns each rank's partial sum and count so the caller can combine them however
    it needs to (a raw dist.all_reduce, or something else entirely for a non-torch.distributed
    caller).
    """
    correct = 0.0
    count = 0
    for idx in range(rank, len(data), world_size):
        is_correct = evaluate_example(idx, model, tokenizer, data, device, task_meta)
        correct += float(is_correct)
        count += 1
    return correct, count
