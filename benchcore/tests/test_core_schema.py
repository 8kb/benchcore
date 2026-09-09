"""
Test benchcore.core's schema scoring path (a Winograd-style task: shared continuation, varying
context) end-to-end against MockTokenizer + ScriptedModel.

python -m pytest benchcore/tests/test_core_schema.py -v
"""
import torch

from benchcore.core import evaluate_task
from benchcore.mock import MockTokenizer, ScriptedModel
from benchcore.prompts import batch_sequences_schema, render_prompts_schema, stack_sequences


def _padded_rows(tokenizer, prompts):
    tokens, _, _ = batch_sequences_schema(tokenizer, prompts)
    pad_id = tokenizer.get_bos_token_id()
    return [tuple(row.tolist()) for row in stack_sequences(tokens, pad_id)]


def _script(model, tokenizer, item, fewshot, delimiter, want_correct):
    prompts = render_prompts_schema(item, delimiter, fewshot)
    rows = _padded_rows(tokenizer, prompts)
    winner_idx = item['gold'] if want_correct else (item['gold'] + 1) % len(rows)
    for i, row in enumerate(rows):
        if i != winner_idx:
            model.mark_wrong(row)


def test_schema_accuracy_is_exact_fraction():
    tokenizer = MockTokenizer()
    model = ScriptedModel(vocab_size=tokenizer.get_vocab_size())
    task_meta = {"task_type": "schema", "num_fewshot": 0, "continuation_delimiter": " "}
    data = [
        {"context_options": ["The cat sat", "The dog sat"], "continuation": "on the mat.", "gold": 0},
        {"context_options": ["He ran fast", "He walked slow"], "continuation": "to the store.", "gold": 1},
        {"context_options": ["She sang loud", "She sang quiet"], "continuation": "in the hall.", "gold": 0},
    ]
    outcomes = [True, False, True]
    for item, want_correct in zip(data, outcomes):
        _script(model, tokenizer, item, [], " ", want_correct)

    correct, count = evaluate_task(model, tokenizer, data, torch.device("cpu"), task_meta)
    assert count == len(data)
    assert correct / count == 2 / 3
