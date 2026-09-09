"""
Test benchcore.core's language_modeling scoring path (batch size 1: does the model's argmax
prediction reproduce the continuation tokens exactly?) end-to-end against MockTokenizer +
ScriptedModel.

python -m pytest benchcore/tests/test_core_lm.py -v
"""
import torch

from benchcore.core import evaluate_task
from benchcore.mock import MockTokenizer, ScriptedModel
from benchcore.prompts import batch_sequences_lm, render_prompts_lm, stack_sequences


def _row(tokenizer, item, fewshot, delimiter):
    prompts = render_prompts_lm(item, delimiter, fewshot)
    tokens, _, _ = batch_sequences_lm(tokenizer, prompts)
    pad_id = tokenizer.get_bos_token_id()
    return tuple(stack_sequences(tokens, pad_id)[0].tolist())


def test_lm_accuracy_is_exact_fraction():
    tokenizer = MockTokenizer()
    model = ScriptedModel(vocab_size=tokenizer.get_vocab_size())
    task_meta = {"task_type": "language_modeling", "num_fewshot": 0, "continuation_delimiter": " "}
    data = [
        {"context": "The chemical formula of water is", "continuation": " H2O."},
        {"context": "The capital of France is", "continuation": " Paris."},
        {"context": "The opposite of hot is", "continuation": " cold."},
    ]
    outcomes = [True, True, False] # exactly 2 of 3 should be predicted correctly
    for item, want_correct in zip(data, outcomes):
        row = _row(tokenizer, item, [], " ")
        if not want_correct:
            model.mark_wrong(row)

    correct, count = evaluate_task(model, tokenizer, data, torch.device("cpu"), task_meta)
    assert count == len(data)
    assert correct / count == 2 / 3
