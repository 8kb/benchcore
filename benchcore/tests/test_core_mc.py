"""
Test benchcore.core's multiple_choice scoring path end-to-end (evaluate_task/evaluate_example)
against MockTokenizer + ScriptedModel: hand-built examples, scripted so the model gets exactly 2
of 3 right, and evaluate_task must report exactly 2/3 accuracy -- no floating-point slop, no
approximate assertion.

python -m pytest benchcore/tests/test_core_mc.py -v
"""
import torch

from benchcore.core import evaluate_task
from benchcore.mock import MockTokenizer, ScriptedModel
from benchcore.prompts import batch_sequences_mc, render_prompts_mc, stack_sequences


def _padded_rows(tokenizer, prompts):
    """The exact token rows ScriptedModel will see, in order -- batch_sequences_mc + the same
    BOS-padding stack_sequences applies before forwarding."""
    tokens, _, _ = batch_sequences_mc(tokenizer, prompts)
    pad_id = tokenizer.get_bos_token_id()
    return [tuple(row.tolist()) for row in stack_sequences(tokens, pad_id)]


def _script(model, tokenizer, item, fewshot, delimiter, want_correct):
    """Marks every row but one BAD so the GOOD row -- the gold choice if want_correct, some other
    choice otherwise -- is the unique minimum-mean-loss row evaluate_example will pick."""
    prompts = render_prompts_mc(item, delimiter, fewshot)
    rows = _padded_rows(tokenizer, prompts)
    winner_idx = item['gold'] if want_correct else (item['gold'] + 1) % len(rows)
    for i, row in enumerate(rows):
        if i != winner_idx:
            model.mark_wrong(row)


def test_multiple_choice_accuracy_is_exact_fraction():
    tokenizer = MockTokenizer()
    model = ScriptedModel(vocab_size=tokenizer.get_vocab_size())
    task_meta = {"task_type": "multiple_choice", "num_fewshot": 0, "continuation_delimiter": " "}
    data = [
        {"query": "The sky is", "choices": ["blue", "green", "red"], "gold": 0},
        {"query": "Grass is", "choices": ["green", "blue"], "gold": 0},
        {"query": "Snow is", "choices": ["white", "black"], "gold": 0},
    ]
    outcomes = [True, True, False] # exactly 2 of 3 should be scored correct
    for item, want_correct in zip(data, outcomes):
        _script(model, tokenizer, item, [], " ", want_correct)

    correct, count = evaluate_task(model, tokenizer, data, torch.device("cpu"), task_meta)
    assert count == len(data)
    assert correct / count == 2 / 3


def test_multiple_choice_all_correct():
    tokenizer = MockTokenizer()
    model = ScriptedModel(vocab_size=tokenizer.get_vocab_size())
    task_meta = {"task_type": "multiple_choice", "num_fewshot": 0, "continuation_delimiter": " "}
    data = [{"query": "1+1=", "choices": ["2", "3"], "gold": 0}]
    _script(model, tokenizer, data[0], [], " ", True)
    correct, count = evaluate_task(model, tokenizer, data, torch.device("cpu"), task_meta)
    assert correct / count == 1.0


def test_multiple_choice_respects_max_seq_len_truncation():
    # a long common-prefix query, short answer choices -- forces truncation to keep the answer
    tokenizer = MockTokenizer()
    long_query = "word " * 50 + "so the sky is"
    item = {"query": long_query, "choices": ["blue", "red"], "gold": 0}
    task_meta = {"task_type": "multiple_choice", "num_fewshot": 0, "continuation_delimiter": " "}

    # figure out how long the tokenized rows are, then pick a max_seq_len that forces cropping
    prompts = render_prompts_mc(item, " ", [])
    tokens, _, _ = batch_sequences_mc(tokenizer, prompts)
    full_len = max(len(t) for t in tokens)
    max_seq_len = full_len - 5
    assert max_seq_len > 0

    model = ScriptedModel(vocab_size=tokenizer.get_vocab_size(), max_seq_len=max_seq_len)
    # script against the CROPPED rows (what the model will actually see)
    cropped_tokens = [t[-max_seq_len:] if len(t) > max_seq_len else t for t in tokens]
    pad_id = tokenizer.get_bos_token_id()
    cropped_rows = [tuple(row.tolist()) for row in stack_sequences(cropped_tokens, pad_id)]
    for i, row in enumerate(cropped_rows):
        if i != item['gold']:
            model.mark_wrong(row)

    correct, count = evaluate_task(model, tokenizer, [item], torch.device("cpu"), task_meta)
    assert correct / count == 1.0
