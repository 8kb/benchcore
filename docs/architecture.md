# benchcore: architecture contract

benchcore scores a model against the CORE benchmark (an in-context-learning suite from the DCLM
paper, https://arxiv.org/abs/2406.11794) and/or a chat-style task suite (ARC/MMLU/GSM8K/
HumanEval). It is the evaluation-side counterpart to `modelcore`/`datacore`: same shape (zero
host-application imports, one Manager entrypoint, its own tests/docs/packaging, an AST guard
proving standalone-ness), different concern — and unlike either of them, it has **zero dependency
on modelcore at all**. A model reaches benchcore through a duck-typed protocol, not a class.

A host application owns checkpoint loading, CLI flags, and the training loop — none of which this
package knows about; its own tokenizer/engine satisfy this package's `Tokenizer`/`Generator`
protocols unmodified. See that host's own `docs/architecture.md` for its side (e.g.
[nanochat's](https://github.com/8kb/nanochat/blob/master/docs/architecture.md#consuming-benchmanager),
whose `scripts/base_eval.py`/`scripts/chat_eval.py` are its CLI entrypoints and
`nanochat.tokenizer.RustBPETokenizer`/`nanochat.engine.Engine` its protocol implementations).

## `BenchManager`: the one entrypoint

Everything a caller needs — score one CORE task, score a whole CORE suite, score one chat task,
score a whole chat suite — goes through `BenchManager`. Nothing else in `benchcore` (`core.py`,
`chat.py`, `prompts.py`, `suite.py` internals) is meant to be reached directly from outside the
package, except the value types and protocols `benchcore/__init__.py` re-exports (`CoreReport`/
`ChatReport`, `CoreSuite`/`CoreTask`/`load_core_suite`, `Task` and its concrete tasks,
`build_chat_tasks`, `Model`/`Tokenizer`/`Generator`, `MockTokenizer`/`ScriptedModel`/
`ScriptedGenerator`).

```python
manager = BenchManager()

# One CORE task, given raw data + task_meta:
accuracy = manager.core_task(model, tokenizer, data, task_meta)

# The whole CORE suite -- core_suite() loads (downloading if needed) and scores in one call:
report = manager.core_suite(model, tokenizer, cache_dir=cache_dir, max_per_task=100)  # report.core_metric
# equivalent to: suite = load_core_suite(cache_dir, max_per_task=100); manager.core(model, tokenizer, suite)

# One chat-style task:
accuracy = manager.chat(task, model, tokenizer, generator=generator)

# The whole ChatCORE suite -- build_chat_tasks() builds the standard {name: Task} dict:
tasks = build_chat_tasks(cache_dir=cache_dir)   # or build_chat_tasks(["ARC-Easy", "MMLU"], cache_dir=cache_dir)
report = manager.chat_suite(tasks, model, tokenizer, generator=generator)
```

`core_suite`/`build_chat_tasks` exist because every caller's own CLI/job-file layer used to
hand-roll the same two things: "load the CORE bundle from a cache dir, then score it" and "map
`ALL_CHAT_TASKS` names to their task classes" (previously duplicated identically across nanochat's
`scripts/base_eval.py`/`scripts/chat_eval.py` and tinylab's `tinylab/ops/bench.py`) — both were
benchcore's own names and classes already, just not benchcore's own function.

## The three protocols

`benchcore.protocols.Model`/`Tokenizer`/`Generator` are `runtime_checkable` `Protocol`s, not ABCs
— benchcore has no business enforcing what a caller's model/tokenizer/generator subclasses from.
Derived directly from what the evaluation code actually calls (not designed up front):

- `Model`: `__call__(input_ids) -> logits`, i.e. exactly what an `nn.Module` already provides.
  `get_device()` and `max_seq_len` are OPTIONAL, checked with `getattr` — same optionality
  convention as `datacore.tokenizer.Tokenizer`'s `token_byte_lengths()`.
- `Tokenizer`: `encode(text, prepend=None)`, `decode(ids)`, `get_bos_token_id()`.
  `render_for_completion(conversation)` is needed only for chat-suite evaluation, not CORE.
- `Generator`: `generate_batch(tokens, num_samples=1, **kwargs) -> (results, masks)` — only
  `results` is read. An OPTIONAL `generate_batch_multi(prompts, num_samples=1, **kwargs)` decodes
  several *different* prompts in one batch, returning the same pair nested one level deeper
  (`results[p]` is prompt `p`'s `num_samples` rows); it is feature-detected with `getattr`, so a
  generator without it keeps working unmodified.

A `modelcore.Model` satisfies `Model` unmodified (it's an `nn.Module`); a host's own tokenizer and
generator commonly satisfy `Tokenizer`/`Generator` unmodified too (`nanochat.tokenizer.
RustBPETokenizer`/`nanochat.engine.Engine`, `tinylab.tokenizer.RustBPETokenizer`/
`tinylab.engine.Engine`). None of them need an adapter class.

## Where a benchmark's "container" half lives vs. its "eval" half

A `Task` here is `datacore.ExampleSet` (sliceable indexable collection, from the sibling repo) plus
exactly two things this repo adds: `eval_type` ('categorical' | 'generative') and `evaluate()`
(plus, for GSM8K, `reward()` for RL). This split is deliberate, not incidental: `ExampleSet`/
`ExampleMixture`/`ExampleSequence` and `load_hub_dataset` have no eval criterion at all and are
useful to a host's own training-data code with zero benchcore dependency (see
[datacore/docs/architecture.md](https://github.com/8kb/datacore/blob/main/docs/architecture.md#examplesethubtable-a-separate-standalone-value-type-surface)).
A benchmark that can be scored belongs in `benchcore/tasks/`; a record collection that can't
(SmolTalk, pure SFT training data) belongs in the host application instead, built on
`datacore.ExampleSet` directly (nanochat's `sft_data.py` and tinylab's `data.py` both do this).

## CORE evaluation

`suite.load_core_suite(cache_dir, max_per_task=None)` downloads (if needed, via
`datacore.download.download_file`) and unzips the DCLM eval bundle, parses `core.yaml` (the task
list: label, ICL type, num_fewshot, continuation_delimiter, dataset URI) and `eval_meta_data.csv`
(each task's random-baseline accuracy), and loads every task's jsonl examples — deterministically
shuffled (seed 1337) and optionally subsampled, matching the original script's behavior exactly.

`core.evaluate_example` renders prompts (`prompts.py`), tokenizes and batches them, forwards the
model, and scores one example according to its ICL type:

- **multiple_choice**: one prompt per choice, sharing a common prefix (the query); the choice
  with the lowest mean loss over its own answer span wins.
- **schema**: one prompt per context option, sharing a common suffix (the continuation); same
  mean-loss-over-span ranking, computed from the *end* of each sequence instead of a shared start.
- **language_modeling**: a single prompt with/without its continuation; correct iff the model's
  argmax prediction reproduces the continuation's tokens exactly, position by position.

`max_seq_len` (an OPTIONAL `Model` attribute) triggers right-aligned truncation with index
shifting, for a model that can't forward beyond a fixed length.

`center(accuracy, random_baseline)` maps `[random_baseline%, 100%]` onto `[0, 1]`; the CORE metric
is the mean of every task's centered accuracy.

`BenchManager.core`/`core_suite` take an optional `log(msg)` callback, called once per completed
task -- CORE's only real progress signal, since `evaluate_task` itself loops examples serially with
no cross-example batching (unlike `chat.run_categorical_eval` below) and has no finer-grained
visibility point worth exposing. A caller with nothing to pass gets identical behavior to before
(`log=None` is the default everywhere it's threaded through).

## Chat-style task evaluation

`chat.run_categorical_eval` batches independent problems (no sampling needed): it renders each
conversation's prompt via `tokenizer.render_for_completion`, forwards the model once per batch,
and picks whichever answer letter has the highest logit at the last prompt position — narrowing
the argmax to just the available letters, which is why every letter must tokenize to exactly one
token (asserted, not just assumed). `chat.run_generative_eval` by default goes one problem at a
time: render the prompt, sample `num_samples` completions from `generator.generate_batch`, and the
problem passes if *any* completion does (pass@k semantics). With `batch_size > 1` and a generator
offering `generate_batch_multi` it decodes that many different problems per batch instead — after
the usual per-rank sharding, sorted by prompt length within a rank, and with each row sliced by its
*own* prompt's length. At temperature 0 the result is identical to the one-at-a-time loop; above 0
one RNG stream now serves a whole batch, so sampled tokens differ. `BenchManager.chat`/`chat_suite`
expose this as `generative_batch_size` (default 1), deliberately separate from `batch_size`, which
is the categorical loop's problems-per-forward and never reaches generation. `eval_workers` scores a
batch's completions in threads (HumanEval runs each in its own subprocess); default 1.

`BenchManager.chat`/`chat_suite` also take an optional `log(msg)`: `chat_suite` calls it once when
each named task starts, and forwards it into `chat()`, which passes it down into whichever loop
runs. `run_generative_eval` and `run_categorical_eval` each call it periodically (about 20 times
over the whole task regardless of task size, via a `max(1, count // 20)` stride) -- the generative
loop is the one that actually needs this: each problem is a full autoregressive decode, the
slowest and most opaque part of the whole suite (GSM8K's ~1,319 problems and HumanEval's ~164 have
each, historically, cost more than training the model under evaluation -- see this family's own
`nanochat/docs/contest.md`).

`chatcore_metric(accuracies, tasks=ALL_CHAT_TASKS, baselines=CHAT_BASELINE_ACCURACIES)` is the
same centering idea as CORE's, applied to a fixed five-task suite (ARC-Easy/ARC-Challenge/MMLU/
GSM8K/HumanEval) whose baselines (`CHAT_BASELINE_ACCURACIES`) are now a single fact instead of a
literal duplicated at every call site (as it was pre-extraction, in both `scripts/chat_eval.py`
and `scripts/chat_sft.py`). `BenchManager.chat_suite` computes it automatically when every name in
`ALL_CHAT_TASKS` was evaluated, and leaves `ChatReport.chatcore_metric` as `None` otherwise — a
caller wanting the metric over some other subset (e.g. just the categorical tasks, as nanochat's
`chat_sft.py` also tracks) calls `chatcore_metric(report.results, tasks=CATEGORICAL_CHAT_TASKS)`
directly.

## No ambient globals, no ambient distributed state

Same rule `modelcore.runtime`/`datacore.DataManager.batches()` both follow, arrived at
independently here: `device`/`cache_dir`/`rank`/`world_size` are always explicit parameters, never
read from an environment variable or a host's own base-directory convention.
`core.evaluate_task`/`chat.run_categorical_eval`/`chat.run_generative_eval` take `rank`/
`world_size` and return each rank's **partial** `(correct, count)` or `(num_passed, total)` rather
than calling `torch.distributed` themselves; `BenchManager` does the actual `all_reduce`, and only
when `world_size > 1`. This is what makes benchcore's own test suite (and any caller with a
different or no distributed setup) free of any `torch.distributed` dependency.

## Sandboxed execution

`execution.execute_code` runs untrusted Python (an LLM's HumanEval completion) in a fresh
subprocess: rlimited memory (best-effort; `resource.setrlimit` is skipped on macOS, see
`GUARD`'s comment), a scrubbed environment, a temp working directory, disabled destructive
builtins (`os.system`, `shutil.rmtree`, `subprocess.Popen`, ...), and a hard kill on timeout. Not a
real security sandbox against adversarial code — see the module docstring's "What is not
covered". `HumanEval.evaluate` is the one caller in this repo; nothing else needs it, but it's
exported (`benchcore.execute_code`) for a host wanting the same sandbox for its own tool-use eval.

## Testing with mocks, no real model

`benchcore.mock.MockTokenizer` is a deterministic char-level tokenizer (mirrors
`datacore.CharTokenizer`'s role for that repo) — guarantees `"A"`/`"B"`/`"C"`/`"D"` are each a
single token, which `run_categorical_eval`'s letter-encoding assertion depends on.

`ScriptedModel` gives exact control over two independent things: `mark_wrong(ids)` flips a whole
row's teacher-forced-roll scoring (used for CORE's mean-loss ranking and LM argmax-match scoring);
`set_next_token(prefix, position, token_id)` forces the logit at one specific (prefix, position)
directly (needed for chat-suite categorical eval, where the predicted token is never actually
part of the input row). `ScriptedGenerator.register(prompt_ids, continuation_ids)` gives the same
exact control over generative eval.

Every scoring path this repo owns (`core.py`'s three ICL types, `chat.py`'s two eval loops, the
CORE and ChatCORE centering math) has a test built on these mocks asserting an *exact* fraction
(e.g. `2/3`), not an approximate or smoke-level check — see `benchcore/tests/`.

## Verifying a change is behavior-preserving

```bash
python -m pytest benchcore/tests -v
```

No real model, no real BPE tokenizer, no network required (except `test_execution.py`'s and
`test_chat_generative.py`'s real sandboxed-subprocess tests, which need no network either — just a
local Python interpreter). For a from-scratch standalone-copy check (the actual proof
`cp -r benchcore /somewhere/else` is a real, testable claim):

```bash
mkdir -p /tmp/bc && cp -r benchcore /tmp/bc/benchcore && cd /tmp/bc && python -m pytest benchcore/tests -v
```

For the CORE prompt-rendering rewrite specifically (jinja2 → plain Python), this repo's own
`tests/test_prompts.py` is independently cross-checked against nanochat's
`tests/goldens/eval_core_prompts.json` (captured from the real pre-extraction jinja2 output by
`dev/capture_eval_goldens.py`, frozen, before this repo existed) — see
[nanochat's docs/architecture.md](https://github.com/8kb/nanochat/blob/master/docs/architecture.md#verifying-a-change-is-behavior-preserving)
for that side. A change to any of the three render functions needs that golden re-verified, not
just this repo's own suite passing. More generally: this repo's own suite proving correctness of a
fresh build is necessary but not proof a change leaves a host unaffected — for a change a host
depends on, also run that host's suite against an editable install
(`uv pip install -e ../benchcore` from its venv). See
[`llmllab/docs/subsystem-conventions.md`](../llmllab/docs/subsystem-conventions.md) for the general
tag-bump/verification contract.
