"""
Test that benchcore.prompts' plain-Python rendering is byte-identical to the jinja2 templates it
replaces (nanochat/core_eval.py's render_prompts_mc/schema/lm, before this repo existed). These
literal expected strings were produced by running the original jinja2 templates against these
exact fixtures (see the benchcore extraction plan's golden-capture step); separately verified
against nanochat's own captured goldens (tests/goldens/eval_core_prompts.json, 5 real examples
across all three task types from the real CORE bundle) at extraction time -- this file's fixtures
are hand-crafted so benchcore's own suite needs no external bundle download.

python -m pytest benchcore/tests/test_prompts.py -v
"""
from benchcore.prompts import (
    batch_sequences_lm,
    batch_sequences_mc,
    batch_sequences_schema,
    find_common_length,
    render_prompts_lm,
    render_prompts_mc,
    render_prompts_schema,
)
from benchcore.mock import MockTokenizer


def test_render_prompts_mc_zeroshot():
    item = {"query": "The sky is", "choices": ["blue", "green", "red"], "gold": 0}
    assert render_prompts_mc(item, " ", []) == [
        "The sky is blue",
        "The sky is green",
        "The sky is red",
    ]


def test_render_prompts_mc_fewshot():
    item = {"query": "The sky is", "choices": ["blue", "green", "red"], "gold": 0}
    fewshot = [
        {"query": "Grass is", "choices": ["green", "blue"], "gold": 0},
        {"query": "Snow is", "choices": ["white", "black"], "gold": 0},
    ]
    assert render_prompts_mc(item, " ", fewshot) == [
        "Grass is green\n\nSnow is white\n\nThe sky is blue",
        "Grass is green\n\nSnow is white\n\nThe sky is green",
        "Grass is green\n\nSnow is white\n\nThe sky is red",
    ]


def test_render_prompts_schema_zeroshot():
    item = {"context_options": ["The cat sat", "The dog sat"], "continuation": "on the mat.", "gold": 0}
    assert render_prompts_schema(item, " ", []) == [
        "The cat sat on the mat.",
        "The dog sat on the mat.",
    ]


def test_render_prompts_schema_fewshot():
    item = {"context_options": ["The cat sat", "The dog sat"], "continuation": "on the mat.", "gold": 0}
    fewshot = [{"context_options": ["He ran fast", "He ran slow"], "gold": 0, "continuation": "to win."}]
    assert render_prompts_schema(item, " ", fewshot) == [
        "He ran fast to win.\n\nThe cat sat on the mat.",
        "He ran fast to win.\n\nThe dog sat on the mat.",
    ]


def test_render_prompts_lm_zeroshot():
    item = {"context": "The chemical formula of water is  ", "continuation": "H2O."}
    assert render_prompts_lm(item, " ", []) == [
        "The chemical formula of water is",
        "The chemical formula of water is H2O.",
    ]


def test_render_prompts_lm_fewshot():
    # deliberately trailing whitespace on both contexts -- must be trimmed, and the final
    # prompt_without must be globally stripped so the delimiter's trailing space doesn't leak
    item = {"context": "The chemical formula of water is  ", "continuation": "H2O."}
    fewshot = [{"context": "The capital of France is ", "continuation": "Paris."}]
    assert render_prompts_lm(item, " ", fewshot) == [
        "The capital of France is Paris.\n\nThe chemical formula of water is",
        "The capital of France is Paris.\n\nThe chemical formula of water is H2O.",
    ]


def test_find_common_length_prefix_and_suffix():
    assert find_common_length([[1, 2, 3], [1, 2, 4]], direction='left') == 2
    assert find_common_length([[1, 2, 3], [9, 2, 3]], direction='right') == 2
    assert find_common_length([[1, 2, 3], [1, 2, 3]], direction='left') == 3


def test_batch_sequences_mc_common_prefix():
    tok = MockTokenizer()
    tokens, start_idxs, end_idxs = batch_sequences_mc(tok, ["The sky is blue", "The sky is red"])
    # start index is where the common prefix ends (after BOS-prepend)
    assert start_idxs[0] == start_idxs[1]
    assert end_idxs[0] == len(tokens[0]) and end_idxs[1] == len(tokens[1])


def test_batch_sequences_schema_common_suffix():
    tok = MockTokenizer()
    tokens, start_idxs, end_idxs = batch_sequences_schema(tok, ["The cat sat on the mat.", "The dog sat on the mat."])
    assert end_idxs[0] == len(tokens[0]) and end_idxs[1] == len(tokens[1])
    # start indices differ since the contexts have different lengths, sharing only the suffix
    assert (end_idxs[0] - start_idxs[0]) == (end_idxs[1] - start_idxs[1])


def test_batch_sequences_lm_prefix_relationship():
    tok = MockTokenizer()
    prompts = render_prompts_lm({"context": "water is", "continuation": " wet"}, " ", [])
    tokens, start_idxs, end_idxs = batch_sequences_lm(tok, prompts)
    assert len(tokens) == 1
    assert start_idxs[0] < end_idxs[0]
