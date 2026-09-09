"""
benchcore -- a standalone evaluation subsystem: score any model against the CORE benchmark (an
ICL suite from the DCLM paper) and/or a chat-style task suite (ARC/MMLU/GSM8K/HumanEval), through
three small duck-typed protocols (Model/Tokenizer/Generator). Knows nothing about a host
application's checkpoint format, CLI flags, or training loop -- see docs/architecture.md for the
full contract, and (in nanochat) scripts/base_eval.py/scripts/chat_eval.py for the layer that
adapts a specific application onto it.

BenchManager is the one entrypoint; CoreReport/ChatReport, CoreSuite/CoreTask/load_core_suite,
Task and its concrete tasks (ARC/MMLU/GSM8K/HumanEval), Model/Tokenizer/Generator, and
MockTokenizer/ScriptedModel/ScriptedGenerator (for a caller's own tests) are the value types that
cross its boundary. Everything else (prompts.py, core.py, chat.py internals) is internal.

Importing this package triggers no side effects.
"""
from benchcore.chat import ALL_CHAT_TASKS, CATEGORICAL_CHAT_TASKS, CHAT_BASELINE_ACCURACIES, chatcore_metric
from benchcore.execution import ExecutionResult, execute_code
from benchcore.manager import BenchManager, ChatReport, CoreReport
from benchcore.mock import MockTokenizer, ScriptedGenerator, ScriptedModel
from benchcore.protocols import Generator, Model, Tokenizer
from benchcore.suite import CoreSuite, CoreTask, center, load_core_suite
from benchcore.tasks import ARC, GSM8K, MMLU, HumanEval, Task, render_mc

__all__ = [
    "BenchManager", "CoreReport", "ChatReport",
    "CoreSuite", "CoreTask", "load_core_suite", "center",
    "Task", "render_mc", "ARC", "MMLU", "GSM8K", "HumanEval",
    "Model", "Tokenizer", "Generator",
    "ExecutionResult", "execute_code",
    "MockTokenizer", "ScriptedModel", "ScriptedGenerator",
    "ALL_CHAT_TASKS", "CATEGORICAL_CHAT_TASKS", "CHAT_BASELINE_ACCURACIES", "chatcore_metric",
]
