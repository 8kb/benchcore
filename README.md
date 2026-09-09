# benchcore

A standalone evaluation subsystem: score any model against the CORE benchmark (an in-context-
learning suite from the DCLM paper) and/or a chat-style task suite (ARC/MMLU/GSM8K/HumanEval).
Zero dependency on any particular model or tokenizer implementation — a model, tokenizer, and
(for generative tasks) a generator each need only satisfy a small duck-typed protocol
(`benchcore.Model`/`Tokenizer`/`Generator`); no benchcore-specific base class required.

It has exactly one public entrypoint, `BenchManager`. It knows nothing about a host application's
checkpoint format, CLI flags, or training loop — that's a host's job, sitting on top and handing
`BenchManager` a loaded model/tokenizer.

See [docs/architecture.md](docs/architecture.md) for the full contract.

## Quickstart

```python
from benchcore import BenchManager, MockTokenizer, ScriptedModel

tokenizer = MockTokenizer()
model = ScriptedModel(vocab_size=tokenizer.get_vocab_size()) # or any real model
manager = BenchManager()

accuracy = manager.core_task(
    model, tokenizer,
    data=[{"query": "The sky is", "choices": ["blue", "green"], "gold": 0}],
    task_meta={"task_type": "multiple_choice", "num_fewshot": 0, "continuation_delimiter": " "},
)
```

Against the real CORE bundle:

```python
from benchcore import BenchManager, load_core_suite

suite = load_core_suite(cache_dir="/path/to/cache", max_per_task=100)
report = BenchManager().core(model, tokenizer, suite)
print(report.core_metric)
```

Chat-style tasks:

```python
from benchcore import ARC, MMLU, GSM8K, HumanEval, BenchManager

tasks = {
    "ARC-Easy": ARC(subset="ARC-Easy", split="test", cache_dir=cache_dir),
    "MMLU": MMLU(subset="all", split="test", cache_dir=cache_dir),
    "GSM8K": GSM8K(subset="main", split="test", cache_dir=cache_dir),
    "HumanEval": HumanEval(cache_dir=cache_dir),
}
report = BenchManager().chat_suite(tasks, model, tokenizer, generator=generator)
print(report.chatcore_metric)
```

## Tests

```bash
python -m pytest benchcore/tests -v
```

No real model, no real BPE tokenizer, no network required — `MockTokenizer`/`ScriptedModel`/
`ScriptedGenerator` (exported for a host's own tests too) cover the whole suite, except
`test_execution.py`'s real sandboxed-subprocess tests and `test_chat_generative.py`'s HumanEval
test, which run the real `execute_code` sandbox. `benchcore/tests/test_standalone.py` mechanically
checks that nothing under `benchcore/` imports a host application (or `modelcore`, a sibling
standalone component — datacore IS an allowed import, since benchcore builds on it); see
[docs/architecture.md](docs/architecture.md#verifying-a-change-is-behavior-preserving) for the
full verification recipe, including a from-scratch standalone-copy check.

## Development

```bash
git clone git@github.com:8kb/benchcore.git && cd benchcore
uv venv && source .venv/bin/activate
uv pip install -e ../datacore  # or let uv sync fetch the pinned tag once one exists
uv sync --group dev --extra bundle
python -m pytest benchcore/tests -v
```

A host application (e.g. [8kb/nanochat](https://github.com/8kb/nanochat)) pins this repo by git
tag in its own `pyproject.toml` (`[tool.uv.sources]`) — `uv sync` there fetches this exact tag.
For the cross-repo inner dev loop, editing both together without round-tripping through a tag:

```bash
# from the host repo, after its own uv sync has run once
uv pip install -e ../benchcore
```

Docs-only changes need no tag bump. A code change should be tagged here, then the host's pin
bumped and its own suite re-run before the change is considered landed — see
[docs/architecture.md#verifying-a-change-is-behavior-preserving](docs/architecture.md#verifying-a-change-is-behavior-preserving)
and [AGENTS.md](AGENTS.md).
