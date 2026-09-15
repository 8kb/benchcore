"""
BenchManager: the one entrypoint. Scores a model (through benchcore.protocols.Model/Tokenizer/
Generator -- any object with the right shape, no benchcore-specific base class required) against
the CORE benchmark (an ICL suite, see suite.py) and/or a chat-style task suite (ARC/MMLU/GSM8K/
HumanEval, see chat.py). Nothing else in benchcore (core.py, chat.py, prompts.py, suite.py
internals) is meant to be reached directly from outside the package, except the value types
benchcore/__init__.py re-exports.

Handles the torch.distributed reduction across ranks itself when world_size > 1 -- core.py/chat.py
stay reduction-agnostic, returning each rank's partial (correct, count) so a caller with a
different distributed setup (or none at all, like benchcore's own tests) never needs
torch.distributed in scope.
"""
from dataclasses import dataclass

import torch

from benchcore.chat import ALL_CHAT_TASKS, chatcore_metric, run_categorical_eval, run_generative_eval
from benchcore.core import evaluate_task
from benchcore.suite import CoreSuite, center, load_core_suite


@dataclass
class CoreReport:
    results: dict # label -> accuracy (0-1)
    centered_results: dict # label -> centered accuracy
    core_metric: float # mean of centered_results.values()


@dataclass
class ChatReport:
    results: dict # task_name -> accuracy (0-1)
    chatcore_metric: float = None # None unless every name in ALL_CHAT_TASKS was evaluated


def _resolve_device(model, device):
    if device is not None:
        return device
    get_device = getattr(model, "get_device", None)
    return get_device() if get_device is not None else torch.device("cpu")


def _reduce_sum(value, world_size):
    """Sums an int/float across ranks via torch.distributed, only when actually distributed --
    keeps a non-distributed caller (including every test in this repo) free of any dependency on
    torch.distributed being initialized at all."""
    if world_size <= 1:
        return value
    import torch.distributed as dist
    t = torch.tensor([value], dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t.item()


class BenchManager:

    # -- CORE (ICL benchmark) --

    def core_task(self, model, tokenizer, data, task_meta, *, device=None, rank=0, world_size=1):
        """Evaluate one CORE-style task's accuracy (0-1) given its raw example list + task_meta
        dict ({'task_type', 'num_fewshot', 'continuation_delimiter'}) -- the building block
        core() uses per task, exposed directly for ad hoc tasks that don't come from a CoreSuite."""
        device = _resolve_device(model, device)
        correct, count = evaluate_task(model, tokenizer, data, device, task_meta, rank=rank, world_size=world_size)
        correct = _reduce_sum(correct, world_size)
        count = _reduce_sum(count, world_size)
        return correct / count

    def core(self, model, tokenizer, suite: CoreSuite, *, device=None, rank=0, world_size=1) -> CoreReport:
        """Evaluate every task in a CoreSuite (see suite.load_core_suite), returning per-task
        accuracy, per-task centered accuracy (against the suite's random baselines), and their
        mean (the CORE metric)."""
        device = _resolve_device(model, device)
        results, centered_results = {}, {}
        for task in suite.tasks:
            accuracy = self.core_task(model, tokenizer, task.data, task.task_meta, device=device, rank=rank, world_size=world_size)
            results[task.label] = accuracy
            centered_results[task.label] = center(accuracy, suite.random_baselines[task.label])
        core_metric = sum(centered_results.values()) / len(centered_results)
        return CoreReport(results=results, centered_results=centered_results, core_metric=core_metric)

    def core_suite(self, model, tokenizer, *, cache_dir, max_per_task=None, device=None, rank=0,
                    world_size=1) -> CoreReport:
        """core(), but also loading (downloading, if needed) the CORE bundle from `cache_dir` --
        moved here from two identical copies (nanochat's scripts/base_eval.py's evaluate_core,
        tinylab's tinylab/ops/bench.py's _run_core), both of which existed only to call
        suite.load_core_suite then self.core(). Printing/CSV-writing stays the caller's own."""
        suite = load_core_suite(cache_dir, max_per_task=max_per_task)
        return self.core(model, tokenizer, suite, device=device, rank=rank, world_size=world_size)

    # -- Chat-style tasks --

    def chat(self, task, model, tokenizer, *, generator=None, batch_size=1, num_samples=1,
              max_new_tokens=512, temperature=0.0, top_k=50, max_problems=None,
              device=None, rank=0, world_size=1) -> float:
        """Evaluate one chat-style Task's accuracy (0-1). `task.eval_type` selects the loop:
        'categorical' reads logits directly (no generator needed); 'generative' samples from
        `generator` and checks the completion (a Generator is required)."""
        if task.eval_type == 'categorical':
            num_passed, total = run_categorical_eval(
                task, tokenizer, model, batch_size, max_problems=max_problems,
                device=device, rank=rank, world_size=world_size,
            )
        elif task.eval_type == 'generative':
            if generator is None:
                raise ValueError(f"task {task!r} is generative and needs a Generator (pass generator=...)")
            num_passed, total = run_generative_eval(
                task, tokenizer, generator, num_samples, max_new_tokens, temperature, top_k,
                max_problems=max_problems, rank=rank, world_size=world_size,
            )
        else:
            raise ValueError(f"Unsupported task eval_type: {task.eval_type}")
        num_passed = _reduce_sum(num_passed, world_size)
        total = _reduce_sum(total, world_size)
        return num_passed / total

    def chat_suite(self, tasks: dict, model, tokenizer, *, generator=None, batch_size=1,
                    num_samples=1, max_new_tokens=512, temperature=0.0, top_k=50,
                    max_problems=None, device=None, rank=0, world_size=1) -> ChatReport:
        """Evaluate a dict of {task_name: Task} in one pass. If every name in
        benchcore.chat.ALL_CHAT_TASKS (ARC-Easy/ARC-Challenge/MMLU/GSM8K/HumanEval) is present,
        also computes the ChatCORE metric (mean centered accuracy against
        benchcore.chat.CHAT_BASELINE_ACCURACIES); otherwise chatcore_metric is None -- a caller
        wanting the metric over a different subset can call benchcore.chat.chatcore_metric(
        report.results, tasks=...) directly."""
        results = {}
        for name, task in tasks.items():
            results[name] = self.chat(
                task, model, tokenizer, generator=generator, batch_size=batch_size,
                num_samples=num_samples, max_new_tokens=max_new_tokens, temperature=temperature,
                top_k=top_k, max_problems=max_problems, device=device, rank=rank, world_size=world_size,
            )
        metric = chatcore_metric(results) if set(ALL_CHAT_TASKS) <= set(results) else None
        return ChatReport(results=results, chatcore_metric=metric)
