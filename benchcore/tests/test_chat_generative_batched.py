"""
run_generative_eval with batch_size > 1: several DIFFERENT problems per decode batch, through a
generator's optional generate_batch_multi. The contract is that batching changes nothing about the
result (same problems, same pass/fail, same sharding) and that everything that doesn't opt in --
default batch_size, a generator without generate_batch_multi -- keeps the original one-at-a-time loop.

python -m pytest benchcore/tests/test_chat_generative_batched.py -v
"""
import pytest

from benchcore import BenchManager
from benchcore.chat import run_generative_eval
from benchcore.mock import MockTokenizer, ScriptedGenerator
from benchcore.tests.test_chat_generative import (
    ToyGSM8KTask, ToyHumanEvalTask, _gsm8k_conversation, _humaneval_conversation,
)

N = 7
RIGHT = {0, 1, 3, 4, 6}  # the problems whose scripted completion is correct; 2 and 5 are wrong


# Problem i's question is LENGTH_RANK[i] * 40 characters longer than the shortest. The order is
# scrambled on purpose: with lengths already ascending by index, sorting by prompt length would be
# a no-op and nothing could notice it going missing.
LENGTH_RANK = [3, 0, 5, 1, 6, 2, 4]


def _problems():
    """N problems whose prompts all differ in length (so every batch is genuinely ragged) and each
    end in a decoy "#### 424242" -- GSM8K scoring regex-searches the completion, so a too-short
    per-row prefix that leaks the tail of the prompt into the completion would score the decoy
    instead of the answer, but only if there is a decoy to leak."""
    conversations = [
        _gsm8k_conversation(f"{'x' * (40 * LENGTH_RANK[i])} What is 1+1? #### 424242", f"#### {i}")
        for i in range(N)
    ]
    return ToyGSM8KTask(conversations), conversations


def _scripted(tokenizer, conversations, generator_cls=ScriptedGenerator):
    generator = generator_cls()
    for i, conversation in enumerate(conversations):
        answer = i if i in RIGHT else 999
        generator.register(tokenizer.render_for_completion(conversation), tokenizer.encode(f"so #### {answer}"))
    return generator


class Recording(ScriptedGenerator):
    """ScriptedGenerator that remembers how it was called."""

    def __init__(self):
        super().__init__()
        self.single_calls, self.multi_calls = [], []

    def generate_batch(self, tokens, num_samples=1, **kwargs):
        self.single_calls.append(list(tokens))
        return super().generate_batch(tokens, num_samples, **kwargs)

    def generate_batch_multi(self, prompts, num_samples=1, **kwargs):
        self.multi_calls.append([list(p) for p in prompts])
        # the base class's own generate_batch_multi routes through self.generate_batch, which this
        # class records -- go around it so single_calls only ever holds calls made by the eval loop
        pairs = [ScriptedGenerator.generate_batch(self, p, num_samples, **kwargs) for p in prompts]
        return [results for results, _ in pairs], [masks for _, masks in pairs]


class SingleOnly:
    """A generator with generate_batch and nothing else -- what any Generator predating
    generate_batch_multi looks like (e.g. a host that never adopted it)."""

    def __init__(self, inner):
        self._inner = inner
        self.calls = 0

    def generate_batch(self, tokens, num_samples=1, **kwargs):
        self.calls += 1
        return self._inner.generate_batch(tokens, num_samples, **kwargs)


def _run(task, tokenizer, generator, **kwargs):
    return run_generative_eval(task, tokenizer, generator, 1, 32, 0.0, None, **kwargs)


def test_batched_result_equals_unbatched_and_uses_multiple_ragged_batches():
    tokenizer = MockTokenizer()
    task, conversations = _problems()
    unbatched = _run(task, tokenizer, _scripted(tokenizer, conversations))
    assert unbatched == (len(RIGHT), N)

    generator = _scripted(tokenizer, conversations, Recording)
    batched = _run(task, tokenizer, generator, batch_size=3)
    assert batched == unbatched
    assert [len(b) for b in generator.multi_calls] == [3, 3, 1]
    assert generator.single_calls == []
    lengths = [len(p) for batch in generator.multi_calls for p in batch]
    assert lengths == sorted(lengths), "problems are sorted by prompt length before being chunked"
    assert all(len({len(p) for p in batch}) == len(batch) for batch in generator.multi_calls), "batches are ragged"


def test_batched_sharding_covers_the_same_problems_as_unbatched_sharding():
    tokenizer = MockTokenizer()
    task, conversations = _problems()
    whole = _run(task, tokenizer, _scripted(tokenizer, conversations))
    seen, passed, total = [], 0, 0
    for rank in (0, 1):
        generator = _scripted(tokenizer, conversations, Recording)
        p, t = _run(task, tokenizer, generator, batch_size=2, rank=rank, world_size=2)
        seen += [tuple(prompt) for batch in generator.multi_calls for prompt in batch]
        passed, total = passed + p, total + t
    assert (passed, total) == whole
    expected = [tuple(tokenizer.render_for_completion(c)) for c in conversations]
    assert sorted(seen) == sorted(expected), "every problem decoded exactly once across ranks"


@pytest.mark.parametrize("max_problems", [1, 2, 5, 6, 7, 50])
def test_batched_respects_max_problems_when_not_a_multiple_of_the_batch(max_problems):
    tokenizer = MockTokenizer()
    task, conversations = _problems()
    unbatched = _run(task, tokenizer, _scripted(tokenizer, conversations), max_problems=max_problems)
    batched = _run(task, tokenizer, _scripted(tokenizer, conversations), max_problems=max_problems, batch_size=3)
    assert batched == unbatched
    assert batched[1] == min(max_problems, N)


def test_a_generator_without_generate_batch_multi_falls_back_to_one_at_a_time():
    tokenizer = MockTokenizer()
    task, conversations = _problems()
    unbatched = _run(task, tokenizer, _scripted(tokenizer, conversations))
    generator = SingleOnly(_scripted(tokenizer, conversations))
    assert _run(task, tokenizer, generator, batch_size=8) == unbatched
    assert generator.calls == N


def test_batching_is_off_by_default_even_for_a_generator_that_supports_it():
    tokenizer = MockTokenizer()
    task, conversations = _problems()
    generator = _scripted(tokenizer, conversations, Recording)
    _run(task, tokenizer, generator)
    assert generator.multi_calls == [] and len(generator.single_calls) == N


class PerSample(ScriptedGenerator):
    """Different continuations per sample, per prompt -- so a bug regrouping a batch's flat
    completion list back into problems (off-by-one at a problem boundary) changes pass/fail."""

    def __init__(self, samples_by_prompt):
        super().__init__()
        self.samples = {tuple(k): v for k, v in samples_by_prompt.items()}

    def generate_batch(self, tokens, num_samples=1, **kwargs):
        rows = [list(tokens) + c for c in self.samples[tuple(tokens)][:num_samples]]
        return rows, [[1] * len(r) for r in rows]


def test_pass_at_k_regroups_each_problems_own_samples_across_a_batch():
    tokenizer = MockTokenizer()
    task, conversations = _problems()
    conversations, task = conversations[:4], ToyGSM8KTask(conversations[:4])
    right = lambda i: tokenizer.encode(f"#### {i}")
    wrong = tokenizer.encode("#### 999")
    # problem 0: right sample is LAST; 1: no right sample; 2: right sample FIRST; 3: none
    samples = {
        tuple(tokenizer.render_for_completion(conversations[0])): [wrong, wrong, right(0)],
        tuple(tokenizer.render_for_completion(conversations[1])): [wrong, wrong, wrong],
        tuple(tokenizer.render_for_completion(conversations[2])): [right(2), wrong, wrong],
        tuple(tokenizer.render_for_completion(conversations[3])): [wrong, wrong, wrong],
    }
    generator = PerSample(samples)
    unbatched = run_generative_eval(task, tokenizer, generator, 3, 32, 0.0, None)
    batched = run_generative_eval(task, tokenizer, generator, 3, 32, 0.0, None, batch_size=4)
    assert unbatched == batched == (2, 4)


@pytest.mark.parametrize("workers", [1, 4])
def test_eval_workers_change_nothing_but_the_threading(workers):
    tokenizer = MockTokenizer()
    task, conversations = _problems()
    expected = _run(task, tokenizer, _scripted(tokenizer, conversations))
    got = _run(task, tokenizer, _scripted(tokenizer, conversations, Recording), batch_size=3, eval_workers=workers)
    assert got == expected


def test_humaneval_through_the_real_sandbox_when_batched():
    tokenizer = MockTokenizer()
    check = "def check(candidate):\n    assert candidate(2, 3) == 5\n"
    conversations = [
        _humaneval_conversation("def add(a, b):\n", "add", check),
        _humaneval_conversation("def add_two_numbers(a, b):\n", "add_two_numbers", check),
        _humaneval_conversation("def plus(a, b):\n    # sum\n", "plus", check),
    ]
    task = ToyHumanEvalTask(conversations)
    bodies = ["    return a + b", "    return a - b", "    return b + a"]  # pass, FAIL, pass
    generator = ScriptedGenerator()
    for conversation, body in zip(conversations, bodies):
        name = conversation["entry_point"]
        generator.register(tokenizer.render_for_completion(conversation),
                           tokenizer.encode(f"```python\ndef {name}(a, b):\n{body}\n```"))
    assert _run(task, tokenizer, generator, batch_size=3, eval_workers=3) == (2, 3)


def test_manager_generative_batch_size_defaults_off_and_is_forwarded_when_set():
    tokenizer = MockTokenizer()
    task, conversations = _problems()
    manager = BenchManager()

    generator = _scripted(tokenizer, conversations, Recording)
    default = manager.chat(task, None, tokenizer, generator=generator, max_new_tokens=32)
    assert generator.multi_calls == [] and default == len(RIGHT) / N

    generator = _scripted(tokenizer, conversations, Recording)
    batched = manager.chat(task, None, tokenizer, generator=generator, max_new_tokens=32,
                           generative_batch_size=4)
    assert [len(b) for b in generator.multi_calls] == [4, 3] and batched == default

    # `batch_size` is the categorical knob and must never switch generative batching on
    generator = _scripted(tokenizer, conversations, Recording)
    manager.chat(task, None, tokenizer, generator=generator, max_new_tokens=32, batch_size=8)
    assert generator.multi_calls == []
