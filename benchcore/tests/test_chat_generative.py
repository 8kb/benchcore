"""
Test benchcore.chat.run_generative_eval against MockTokenizer + ScriptedGenerator: GSM8K's
"#### answer" extraction pass/fail, and HumanEval through the REAL execute_code sandbox (one
passing, one failing program) -- not mocked, since the sandbox subprocess mechanism is exactly
what's under test there.

python -m pytest benchcore/tests/test_chat_generative.py -v
"""
from benchcore.chat import run_generative_eval
from benchcore.mock import MockTokenizer, ScriptedGenerator
from benchcore.tasks.gsm8k import extract_answer
from benchcore.tasks.humaneval import HumanEval


class ToyGSM8KTask:
    """Mirrors benchcore.tasks.GSM8K's evaluate() (#### extraction) over hand-built conversations,
    without needing the real HF dataset."""

    eval_type = 'generative'

    def __init__(self, conversations):
        self.conversations = conversations

    def __len__(self):
        return len(self.conversations)

    def __getitem__(self, index):
        return self.conversations[index]

    def evaluate(self, conversation, completion):
        last_text_part = conversation['messages'][-1]['content'][-1]['text']
        return int(extract_answer(completion) == extract_answer(last_text_part))


def _gsm8k_conversation(question, answer_text):
    return {
        "messages": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": [{"type": "text", "text": answer_text}]},
        ],
    }


def test_generative_eval_gsm8k_extraction_exact_fraction():
    tokenizer = MockTokenizer()
    generator = ScriptedGenerator()

    conversations = [
        _gsm8k_conversation("2+2?", "The answer is #### 4"),
        _gsm8k_conversation("3+3?", "The answer is #### 6"),
        _gsm8k_conversation("4+4?", "The answer is #### 8"),
    ]
    task = ToyGSM8KTask(conversations)
    completions = ["Reasoning... #### 4", "Reasoning... #### 6", "Reasoning... #### 99"] # last one wrong
    for conversation, completion_text in zip(conversations, completions):
        prompt_ids = tokenizer.render_for_completion(conversation)
        generator.register(prompt_ids, tokenizer.encode(completion_text))

    num_passed, total = run_generative_eval(task, tokenizer, generator, num_samples=1, max_new_tokens=32, temperature=0.0, top_k=None)
    assert total == 3
    assert num_passed == 2


def test_generative_eval_num_samples_any_pass():
    # with num_samples>1, a single passing sample is enough (pass@k semantics)
    tokenizer = MockTokenizer()
    generator = ScriptedGenerator()
    conversation = _gsm8k_conversation("1+1?", "#### 2")
    task = ToyGSM8KTask([conversation])
    prompt_ids = tokenizer.render_for_completion(conversation)
    generator.register(prompt_ids, tokenizer.encode("#### 2")) # ScriptedGenerator always returns this
    num_passed, total = run_generative_eval(task, tokenizer, generator, num_samples=4, max_new_tokens=8, temperature=1.0, top_k=None)
    assert num_passed == 1 and total == 1


def _humaneval_conversation(prompt, entry_point, test):
    return {
        "messages": [{"role": "user", "content": prompt}],
        "entry_point": entry_point,
        "test": test,
    }


class ToyHumanEvalTask:
    """Reuses HumanEval.evaluate() (which never touches self) over hand-built conversations, so
    this test exercises the real sandboxed execute_code path without a real HF dataset download."""

    eval_type = 'generative'

    def __init__(self, conversations):
        self.conversations = conversations

    def __len__(self):
        return len(self.conversations)

    def __getitem__(self, index):
        return self.conversations[index]

    def evaluate(self, conversation, completion):
        return HumanEval.evaluate(self, conversation, completion)


def test_generative_eval_humaneval_real_sandbox():
    tokenizer = MockTokenizer()
    generator = ScriptedGenerator()

    conversations = [
        _humaneval_conversation(
            prompt="def add(a, b):\n",
            entry_point="add",
            test="def check(candidate):\n    assert candidate(2, 3) == 5\n",
        ),
        _humaneval_conversation(
            # a distinct prompt so ScriptedGenerator (keyed by exact prompt tokens) doesn't
            # collide the two conversations' registered continuations
            prompt="def add_two_numbers(a, b):\n",
            entry_point="add_two_numbers",
            test="def check(candidate):\n    assert candidate(2, 3) == 5\n",
        ),
    ]
    task = ToyHumanEvalTask(conversations)

    completions = [
        "```python\ndef add(a, b):\n    return a + b\n```", # correct
        "```python\ndef add_two_numbers(a, b):\n    return a - b\n```", # wrong -- fails the real sandboxed assertion
    ]
    for conversation, completion_text in zip(conversations, completions):
        prompt_ids = tokenizer.render_for_completion(conversation)
        generator.register(prompt_ids, tokenizer.encode(completion_text))

    num_passed, total = run_generative_eval(task, tokenizer, generator, num_samples=1, max_new_tokens=32, temperature=0.0, top_k=None)
    assert total == 2
    assert num_passed == 1
