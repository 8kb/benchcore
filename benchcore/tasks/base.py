"""
Task: a benchmark's ExampleSet, adding the eval-specific half nanochat/tasks/common.py's Task
class used to carry (eval_type, evaluate(), reward()) on top of datacore.ExampleSet's plain
indexable-slicing container. Also render_mc, the shared multiple-choice rendering convention every
categorical task here uses.

The container half (slicing, ExampleMixture/ExampleSequence, HubTable/load_hub_dataset) lives in
datacore -- it has no eval criterion at all, so it's not benchcore's concern. See
datacore/docs/architecture.md for that split's rationale.
"""
from datacore import ExampleSet


class Task(ExampleSet):
    """
    Base class of a benchmark Task. A Task is a dataset of conversations, together with an
    evaluation criterion. Example tasks: MMLU, ARC-Easy, ARC-Challenge, GSM8K, HumanEval.
    """

    @property
    def eval_type(self):
        # one of 'generative' | 'categorical'
        raise NotImplementedError

    def get_example(self, index):
        raise NotImplementedError

    def evaluate(self, conversation, completion):
        raise NotImplementedError


def render_mc(question, letters, choices):
    """
    The common multiple choice rendering format used across categorical tasks.

    Note two important design decisions:
    1)
    Bigger models don't care as much, but smaller models prefer to have
    the letter *after* the choice, which results in better binding.
    2)
    There is no whitespace between the delimiter (=) and the letter.
    This is actually critical because the tokenizer has different token ids
    for " A" vs. "A". The assistant responses will be just the letter itself,
    i.e. "A", so it is important that here in the prompt it is the exact same
    token, i.e. "A" with no whitespace before it. Again, bigger models don't care
    about this too much, but smaller models do care about some of these details.
    """
    query = f"Multiple Choice question: {question}\n"
    query += "".join([f"- {choice}={letter}\n" for letter, choice in zip(letters, choices)])
    query += "\nRespond only with the letter of the correct answer."
    return query
