"""
Test benchcore.chat.chatcore_metric (mean centered accuracy against hand-computed values) and
BenchManager.chat_suite's decision of when to compute it -- only when every name in
benchcore.chat.ALL_CHAT_TASKS was actually evaluated.

python -m pytest benchcore/tests/test_chat_metric.py -v
"""
import pytest

from benchcore.chat import ALL_CHAT_TASKS, CATEGORICAL_CHAT_TASKS, CHAT_BASELINE_ACCURACIES, build_chat_tasks, chatcore_metric
from benchcore.manager import BenchManager
from benchcore.mock import MockTokenizer, ScriptedGenerator, ScriptedModel
from benchcore.tasks.base import Task


def test_chatcore_metric_hand_computed():
    accuracies = {
        'ARC-Easy': 0.5,       # baseline 0.25 -> centered (0.5-0.25)/(1-0.25) = 1/3
        'ARC-Challenge': 0.25, # baseline 0.25 -> centered 0.0
        'MMLU': 1.0,           # baseline 0.25 -> centered 1.0
        'GSM8K': 0.5,          # baseline 0.0  -> centered 0.5
        'HumanEval': 0.0,      # baseline 0.0  -> centered 0.0
    }
    expected = (1 / 3 + 0.0 + 1.0 + 0.5 + 0.0) / 5
    assert abs(chatcore_metric(accuracies) - expected) < 1e-9


def test_chatcore_metric_over_categorical_subset():
    accuracies = {'ARC-Easy': 1.0, 'ARC-Challenge': 0.25, 'MMLU': 0.625}
    expected = (1.0 + 0.0 + 0.5) / 3
    result = chatcore_metric(accuracies, tasks=CATEGORICAL_CHAT_TASKS)
    assert abs(result - expected) < 1e-9


def test_perfect_and_baseline_accuracy_bound_the_metric():
    perfect = {t: 1.0 for t in ALL_CHAT_TASKS}
    assert chatcore_metric(perfect) == 1.0
    at_baseline = {t: CHAT_BASELINE_ACCURACIES[t] for t in ALL_CHAT_TASKS}
    assert abs(chatcore_metric(at_baseline)) < 1e-9


class AlwaysCorrectCategoricalTask(Task):
    """A trivial categorical task whose evaluate() ignores the prediction entirely -- exists only
    to exercise BenchManager.chat_suite's wiring/metric-inclusion decision, not run_categorical_eval's
    scoring logic (that's test_chat_categorical.py's job)."""

    eval_type = 'categorical'

    def num_examples(self):
        return 2

    def get_example(self, index):
        return {"messages": [{"role": "user", "content": f"Q{index}"}], "letters": ("A", "B")}

    def evaluate(self, conversation, predicted_letter):
        return True


class AlwaysCorrectGenerativeTask(Task):
    """Same idea, for the generative loop."""

    eval_type = 'generative'

    def num_examples(self):
        return 2

    def get_example(self, index):
        return {"messages": [{"role": "user", "content": f"Q{index}"}, {"role": "assistant", "content": "ignored"}]}

    def evaluate(self, conversation, completion):
        return True


def test_chat_suite_computes_metric_when_all_tasks_present():
    tokenizer = MockTokenizer()
    model = ScriptedModel(vocab_size=tokenizer.get_vocab_size())
    generator = ScriptedGenerator()
    tasks = {name: (AlwaysCorrectCategoricalTask() if name in CATEGORICAL_CHAT_TASKS else AlwaysCorrectGenerativeTask())
             for name in ALL_CHAT_TASKS}

    report = BenchManager().chat_suite(tasks, model, tokenizer, generator=generator)

    assert all(acc == 1.0 for acc in report.results.values())
    assert report.chatcore_metric == 1.0


def test_chat_suite_omits_metric_for_a_partial_task_set():
    tokenizer = MockTokenizer()
    model = ScriptedModel(vocab_size=tokenizer.get_vocab_size())
    generator = ScriptedGenerator()
    tasks = {"ARC-Easy": AlwaysCorrectCategoricalTask(), "MMLU": AlwaysCorrectCategoricalTask()}

    report = BenchManager().chat_suite(tasks, model, tokenizer, generator=generator)

    assert set(report.results) == {"ARC-Easy", "MMLU"}
    assert report.chatcore_metric is None


def test_build_chat_tasks_default_covers_every_name(monkeypatch):
    """build_chat_tasks(names=None) builds one of each of the five ALL_CHAT_TASKS, with the right
    constructor kwargs -- monkeypatched to sentinel classes so this needs no network/cache_dir
    (a real ARC/MMLU/GSM8K/HumanEval loads data eagerly in __init__)."""
    calls = []

    def _sentinel(name):
        class _Sentinel:
            def __init__(self, **kwargs):
                calls.append((name, kwargs))
        return _Sentinel

    monkeypatch.setattr("benchcore.tasks.ARC", _sentinel("ARC"))
    monkeypatch.setattr("benchcore.tasks.MMLU", _sentinel("MMLU"))
    monkeypatch.setattr("benchcore.tasks.GSM8K", _sentinel("GSM8K"))
    monkeypatch.setattr("benchcore.tasks.HumanEval", _sentinel("HumanEval"))

    tasks = build_chat_tasks(cache_dir="/tmp/whatever")
    assert set(tasks) == set(ALL_CHAT_TASKS)
    called_names = [name for name, _ in calls]
    assert called_names.count("ARC") == 2       # ARC-Easy + ARC-Challenge
    assert called_names.count("MMLU") == 1
    assert called_names.count("GSM8K") == 1
    assert called_names.count("HumanEval") == 1
    for _, kwargs in calls:
        assert kwargs.get("cache_dir") == "/tmp/whatever"


def test_build_chat_tasks_selects_a_subset(monkeypatch):
    monkeypatch.setattr("benchcore.tasks.ARC", lambda **kw: object())
    monkeypatch.setattr("benchcore.tasks.MMLU", lambda **kw: object())

    tasks = build_chat_tasks(["ARC-Easy", "MMLU"], cache_dir="/tmp/whatever")
    assert set(tasks) == {"ARC-Easy", "MMLU"}


def test_build_chat_tasks_rejects_unknown_name():
    with pytest.raises(ValueError, match="Unknown chat task"):
        build_chat_tasks(["NotATask"], cache_dir="/tmp/whatever")
