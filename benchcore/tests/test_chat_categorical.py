"""
Test benchcore.chat.run_categorical_eval (ARC/MMLU-shaped: pick the answer letter with the
highest logit, no sampling) against MockTokenizer + ScriptedModel, including mixed-length prompts
in one batch (padding).

python -m pytest benchcore/tests/test_chat_categorical.py -v
"""
from benchcore.chat import run_categorical_eval
from benchcore.mock import MockTokenizer, ScriptedModel
from benchcore.tasks.base import Task


class ToyCategoricalTask(Task):
    """A tiny in-memory categorical Task: conversation[-1] is the gold answer letter."""

    def __init__(self, conversations, **kwargs):
        super().__init__(**kwargs)
        self.conversations = conversations

    @property
    def eval_type(self):
        return 'categorical'

    def num_examples(self):
        return len(self.conversations)

    def get_example(self, index):
        return self.conversations[index]

    def evaluate(self, conversation, predicted_letter):
        gold_letter = conversation['messages'][-1]['content']
        return predicted_letter == gold_letter


def _conversation(question, letters, gold_letter):
    return {
        "messages": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": gold_letter},
        ],
        "letters": letters,
    }


def test_categorical_eval_exact_fraction_with_mixed_length_prompts():
    tokenizer = MockTokenizer()
    model = ScriptedModel(vocab_size=tokenizer.get_vocab_size())

    conversations = [
        _conversation("Q1?", ("A", "B"), "A"),
        _conversation("A much longer question that pads the batch out?", ("A", "B", "C"), "B"),
        _conversation("Q3?", ("A", "B"), "A"),
    ]
    task = ToyCategoricalTask(conversations)
    # script the model to answer correctly on examples 0 and 1, incorrectly on example 2
    desired_answers = ["A", "B", "B"] # example 2's gold is "A", model will say "B" -- wrong
    for conversation, answer_letter in zip(conversations, desired_answers):
        prompt_ids = tokenizer.render_for_completion(conversation)
        letter_id = tokenizer.encode(answer_letter)[0]
        model.set_next_token(prompt_ids, position=len(prompt_ids) - 1, token_id=letter_id)

    num_passed, total = run_categorical_eval(task, tokenizer, model, batch_size=2)
    assert total == 3
    assert num_passed == 2


def test_categorical_eval_respects_max_problems():
    tokenizer = MockTokenizer()
    model = ScriptedModel(vocab_size=tokenizer.get_vocab_size())
    conversations = [_conversation(f"Q{i}?", ("A", "B"), "A") for i in range(5)]
    task = ToyCategoricalTask(conversations)
    for conversation in conversations:
        prompt_ids = tokenizer.render_for_completion(conversation)
        letter_id = tokenizer.encode("A")[0]
        model.set_next_token(prompt_ids, position=len(prompt_ids) - 1, token_id=letter_id)

    num_passed, total = run_categorical_eval(task, tokenizer, model, batch_size=2, max_problems=3)
    assert total == 3
    assert num_passed == 3


def test_categorical_eval_shards_across_ranks():
    tokenizer = MockTokenizer()
    model = ScriptedModel(vocab_size=tokenizer.get_vocab_size())
    conversations = [_conversation(f"Q{i}?", ("A", "B"), "A") for i in range(4)]
    task = ToyCategoricalTask(conversations)
    for conversation in conversations:
        prompt_ids = tokenizer.render_for_completion(conversation)
        letter_id = tokenizer.encode("A")[0]
        model.set_next_token(prompt_ids, position=len(prompt_ids) - 1, token_id=letter_id)

    # batch_size=1 -> 4 batches, sharded 2 ranks/2 batches each
    passed0, total0 = run_categorical_eval(task, tokenizer, model, batch_size=1, rank=0, world_size=2)
    passed1, total1 = run_categorical_eval(task, tokenizer, model, batch_size=1, rank=1, world_size=2)
    assert total0 + total1 == 4
    assert passed0 + passed1 == 4
