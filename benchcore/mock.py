"""
Mock model/tokenizer/generator for benchcore's own test suite -- no real model, no real BPE
tokenizer, no network. Modelled on datacore.CharTokenizer: a tiny, dependency-free stand-in that
satisfies benchcore.protocols.{Model,Tokenizer,Generator} exactly, so the same evaluation code
paths (core.py, chat.py) that run against a real modelcore.Model/nanochat.tokenizer/nanochat.engine
get exercised deterministically.
"""
import torch


class MockTokenizer:
    """
    Deterministic char-level tokenizer. Every character in `chars` (default: space, newline, and
    printable ASCII) maps to its own single token id -- in particular this guarantees "A"/"B"/"C"/
    "D" are each exactly one token, which run_categorical_eval's letter-encoding asserts on.
    """

    def __init__(self, chars=None):
        chars = chars if chars is not None else (" \n" + "".join(chr(c) for c in range(32, 127)))
        seen = set()
        alphabet = []
        for c in chars:
            if c not in seen:
                seen.add(c)
                alphabet.append(c)
        self._char_to_id = {c: i for i, c in enumerate(alphabet)}
        self._id_to_char = {i: c for i, c in enumerate(alphabet)}
        self._bos_id = len(alphabet) # BOS gets the id right after the alphabet

    def get_vocab_size(self):
        return self._bos_id + 1

    def get_bos_token_id(self):
        return self._bos_id

    def _encode_one(self, text, prepend_id):
        ids = [] if prepend_id is None else [prepend_id]
        ids.extend(self._char_to_id[c] for c in text)
        return ids

    def encode(self, text, prepend=None):
        if prepend is not None and not isinstance(prepend, int):
            raise ValueError(f"MockTokenizer only supports an int prepend id, got {prepend!r}")
        if isinstance(text, str):
            return self._encode_one(text, prepend)
        return [self._encode_one(t, prepend) for t in text]

    def __call__(self, *args, **kwargs):
        return self.encode(*args, **kwargs)

    def decode(self, ids):
        return "".join(self._id_to_char.get(i, "") for i in ids)

    def render_for_completion(self, conversation):
        """Renders every message except the last (assumed to be the withheld ground-truth
        assistant answer) as 'role: content\\n', BOS-prepended. Deliberately simple -- not a real
        chat template, just enough structure for ScriptedModel/ScriptedGenerator to recognize
        which conversation produced a given prompt."""
        messages = conversation["messages"]
        prompt_messages = messages[:-1] if len(messages) > 1 else messages
        text = "".join(
            f"{m['role']}: {m['content']}\n" for m in prompt_messages if isinstance(m["content"], str)
        )
        return self.encode(text, prepend=self.get_bos_token_id())


class ScriptedModel:
    """
    A deterministic Model for benchcore's own tests. By default, every row forwarded to it is
    scored as GOOD: at each position, the logits put an overwhelming spike on that row's own
    actual next token (from teacher-forcing's roll), so the loss is ~0 everywhere and the argmax
    prediction always matches the target. Call mark_wrong(ids) with the *exact* token ids that
    will actually be forwarded (i.e. already truncated to max_seq_len, if that applies) to flip a
    specific row to BAD: the spike lands on a decoy token instead, guaranteeing both a high loss
    and an argmax mismatch at every position.

    This makes both benchcore.core.py code paths exactly predictable: for multiple_choice/schema,
    marking every wrong answer choice's row BAD (and leaving the correct one GOOD) makes the
    correct choice the unique minimum-mean-loss row; for language_modeling, a GOOD row is scored
    correct (argmax matches throughout), a BAD one incorrect.

    A separate mechanism, set_next_token(), targets a single (prefix, position) directly rather
    than a whole row's teacher-forced roll -- needed by chat-suite categorical eval, where the
    token being predicted (an answer letter) is never part of the input row at all.
    """

    def __init__(self, vocab_size=128, max_seq_len=None, spike=50.0):
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.spike = spike
        self._bad_rows = set()
        self._overrides = {} # (prefix_tuple, position) -> token_id

    def mark_wrong(self, ids):
        self._bad_rows.add(tuple(int(i) for i in ids))

    def set_next_token(self, prefix_ids, position, token_id):
        """Force the logits at `position` to spike on `token_id`, for any row whose first
        `position + 1` tokens equal `prefix_ids` -- decoupled from batch padding, since padding
        only ever appends tokens after `position`, never changes it."""
        self._overrides[(tuple(int(i) for i in prefix_ids), position)] = token_id

    def get_device(self):
        return torch.device("cpu")

    def __call__(self, input_ids):
        batch_size, seq_len = input_ids.shape
        logits = torch.zeros(batch_size, seq_len, self.vocab_size)
        target_ids = torch.roll(input_ids, shifts=-1, dims=1)
        for b in range(batch_size):
            row = [int(i) for i in input_ids[b].tolist()]
            is_bad = tuple(row) in self._bad_rows
            for t in range(seq_len - 1):
                target = int(target_ids[b, t].item())
                spike_id = ((target + 1) % self.vocab_size) if is_bad else target
                logits[b, t, spike_id] = self.spike
            for (prefix, pos), token_id in self._overrides.items():
                if pos < seq_len and tuple(row[:pos + 1]) == prefix:
                    logits[b, pos, :] = 0.0
                    logits[b, pos, token_id] = self.spike
        return logits


class ScriptedGenerator:
    """
    A deterministic Generator for benchcore's own tests: generate_batch(tokens, num_samples, ...)
    returns num_samples copies of `tokens + continuation`, where continuation is looked up by the
    exact prompt token tuple (see register()). An unregistered prompt returns no continuation
    (the completion decodes to an empty string). generate_batch_multi is the same lookup per
    prompt, so this generator advertises the optional multi-prompt capability -- wrap it (or
    subclass and delete the method) to test a generator that doesn't.
    """

    def __init__(self):
        self._continuations = {}

    def register(self, prompt_ids, continuation_ids):
        self._continuations[tuple(prompt_ids)] = list(continuation_ids)

    def generate_batch(self, tokens, num_samples=1, **kwargs):
        continuation = self._continuations.get(tuple(tokens), [])
        result = list(tokens) + continuation
        results = [list(result) for _ in range(num_samples)]
        masks = [[1] * len(result) for _ in range(num_samples)]
        return results, masks

    def generate_batch_multi(self, prompts, num_samples=1, **kwargs):
        pairs = [self.generate_batch(prompt, num_samples, **kwargs) for prompt in prompts]
        return [results for results, _ in pairs], [masks for _, masks in pairs]
