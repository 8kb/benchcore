# AGENTS.md

`benchcore` is a standalone evaluation subsystem — scores any model against the CORE benchmark
(an ICL suite from the DCLM paper) and/or a chat-style task suite (ARC/MMLU/GSM8K/HumanEval)
through three small duck-typed protocols (`Model`/`Tokenizer`/`Generator`). Read
[docs/architecture.md](docs/architecture.md) for the full contract before touching anything here.
For the family-wide pattern this repo follows (one entrypoint, zero host imports, the tag-pin
consumption contract) see [llmllab/AGENTS.md](../llmllab/AGENTS.md) and
[llmllab/docs/subsystem-conventions.md](../llmllab/docs/subsystem-conventions.md).

A host application pins this repo by git tag (`pyproject.toml`'s `[tool.uv.sources]`) and consumes
it entirely through `BenchManager` — see [`llmllab/AGENTS.md`](../llmllab/AGENTS.md)'s family map
for which repos currently do that, and each one's own `docs/architecture.md` for its side of the
contract.

benchcore builds on [8kb/datacore](https://github.com/8kb/datacore) (`ExampleSet`,
`load_hub_dataset`) — the only standalone sibling it depends on. It does **not** depend on
`modelcore`: a `benchcore.Model` is any object with a `__call__(input_ids) -> logits` (plus two
optional members), so a modelcore `Model`, a raw HuggingFace model, or a mock all work unmodified.

## Repo map

```
benchcore/
├── manager.py       BenchManager -- the one entrypoint (core / core_task / chat / chat_suite)
├── protocols.py     Model / Tokenizer / Generator -- the duck-typed protocols a caller's
│                    model/tokenizer/generator must satisfy
├── prompts.py       CORE prompt rendering (plain Python, no jinja2) + batch_sequences_mc/schema/lm
├── core.py            forward_model, evaluate_example, evaluate_task -- the CORE scoring loop
├── suite.py             CORE bundle: download/parse core.yaml + eval_meta_data.csv, centering math
├── chat.py             generative + categorical chat-eval loops, ChatCORE metric + baselines
├── execution.py         sandboxed execute_code, for HumanEval (and any future tool-use eval)
├── tasks/                base.py (Task(datacore.ExampleSet) + eval_type/evaluate + render_mc),
│                        arc.py, mmlu.py, gsm8k.py, humaneval.py
├── mock.py               MockTokenizer, ScriptedModel, ScriptedGenerator -- for a caller's own
│                        tests, and for this repo's own test suite
└── tests/                 this repo's own test suite, incl. test_standalone.py
```

## Invariants that will bite you

- **No ambient distributed state.** `core.evaluate_task` and both `chat.py` eval loops take
  explicit `rank`/`world_size` parameters and return each rank's partial `(correct, count)` /
  `(num_passed, total)` rather than calling `torch.distributed` themselves — mirrors
  `datacore.DataManager.batches()`'s `rank`/`world_size`/`device` parameters (see
  [datacore/AGENTS.md](../datacore/AGENTS.md)). `BenchManager` does the actual `dist.all_reduce`,
  and only when `world_size > 1` — a non-distributed caller (including this repo's entire test
  suite) never needs `torch.distributed` initialized at all.
- **`device`/`cache_dir` are always explicit, never read from an ambient global.** `BenchManager`
  resolves a missing `device` via `getattr(model, "get_device", ...)`, never an environment
  variable or a host's own base-directory convention; `load_core_suite`/`load_hub_dataset` both
  take `cache_dir` as a required parameter.
- **jinja2 is gone; `prompts.py`'s plain-Python rendering must stay byte-identical to what it
  replaced.** The three templates (`render_prompts_mc/schema/lm`) look mechanical but their
  whitespace handling was jinja2 whitespace-control-tag-derived — see `prompts.py`'s module
  docstring and `tests/test_prompts.py`'s literal expected strings (independently cross-checked
  against nanochat's `tests/goldens/eval_core_prompts.json`, captured from the real jinja2 output
  before this repo existed). A change to any of the three render functions needs that golden
  re-verified, not just this repo's own suite passing.
- **`Task` (this repo) is not `datacore.ExampleSet` (the sibling repo) with eval bolted on
  casually — it's a deliberate split.** `ExampleSet`/`ExampleMixture`/`ExampleSequence` and
  `load_hub_dataset` own the container half (slicing, deterministic mixing, HF-hub parquet read);
  `Task` adds only `eval_type`/`evaluate()`/`reward()` and `render_mc`. A new benchmark task
  belongs here (subclassing `Task`); a new *training-data* container with no eval criterion
  belongs in the host application instead (nanochat's `sft_data.py` and tinylab's `data.py` both
  build their own SmolTalk container this way) — see
  [datacore/docs/architecture.md#examplesethubtable-a-separate-standalone-value-type-surface](https://github.com/8kb/datacore/blob/main/docs/architecture.md#examplesethubtable-a-separate-standalone-value-type-surface).
- **`ScriptedModel.mark_wrong`/`set_next_token` are two independent mechanisms, not one.**
  `mark_wrong(ids)` flips a whole row's teacher-forced roll (used by CORE's mean-loss ranking and
  LM argmax-match scoring); `set_next_token(prefix, position, token_id)` targets a single
  (prefix, position) directly (needed by chat-suite categorical eval, where the token being
  predicted — an answer letter — is never part of the input row at all). Conflating them, or
  assuming one subsumes the other, will silently mis-score a test.

## Testing

```bash
python -m pytest benchcore/tests -v
```

No real model, no real BPE tokenizer, no network required for the vast majority of the suite —
see [README.md](README.md#tests) for what the two exceptions (`test_execution.py`,
`test_chat_generative.py`'s HumanEval test) actually exercise for real, and why.
`benchcore/tests/test_standalone.py` mechanically checks that nothing under `benchcore/` imports a
host application (or `modelcore`) — see
[docs/architecture.md#verifying-a-change-is-behavior-preserving](docs/architecture.md#verifying-a-change-is-behavior-preserving)
for the from-scratch standalone-copy recipe.

A change here that a host application depends on needs that host's own suite run against it too,
after an editable install (`uv pip install -e ../benchcore` from the host's venv) — see
[`llmllab/docs/subsystem-conventions.md`](../llmllab/docs/subsystem-conventions.md)'s tag-bump rule.
This repo's own tests proving *it* still works is necessary but not sufficient proof a host is
unaffected.

**Known pre-existing failure, not a regression, on macOS specifically:**
`test_execution.py::test_memory_limit` fails there — the guard's `resource.setrlimit` calls are
skipped on darwin (see `execution.py`'s `GUARD`), so the 256MB limit isn't actually enforced on
that platform. Pre-existing before this code moved into its own repo; it followed the code
unchanged.

## Style

See [llmllab/AGENTS.md](../llmllab/AGENTS.md#style).
