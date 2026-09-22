"""
Model/Tokenizer/Generator: the duck-typed protocols benchcore needs from a host application's
model and tokenizer. Not ABCs -- benchcore has no business enforcing what a caller's model or
tokenizer subclasses from; any object with the right shape works unmodified. Derived directly from
what nanochat/core_eval.py and scripts/chat_eval.py actually called before this package existed --
nanochat.checkpoint_manager's loaded model, nanochat.tokenizer.RustBPETokenizer, and
nanochat.engine.Engine all satisfy these unmodified, with zero adapter code.

`get_device()`/`max_seq_len` are OPTIONAL members of Model, checked with `getattr` rather than
declared on the Protocol -- same optionality convention as datacore.tokenizer.Tokenizer's
token_byte_lengths().
"""
from typing import Protocol, runtime_checkable


@runtime_checkable
class Model(Protocol):
    def __call__(self, input_ids):
        """input_ids: (B, T) int64 tensor of token ids. Returns (B, T, vocab_size) logits."""
        ...

    # OPTIONAL, not declared here (see module docstring):
    #   get_device() -> torch.device
    #   max_seq_len: int | None  (a model that can't forward beyond a fixed length, e.g. GPT-2)


@runtime_checkable
class Tokenizer(Protocol):
    def encode(self, text, prepend=None):
        """text: str or list[str]. Returns list[int] or list[list[int]] to match."""
        ...

    def decode(self, ids) -> str:
        ...

    def get_bos_token_id(self) -> int:
        ...

    # OPTIONAL, not declared here: render_for_completion(conversation) -> list[int], needed only
    # by chat-suite evaluation (BenchManager.chat/chat_suite), not by CORE.


@runtime_checkable
class Generator(Protocol):
    def generate_batch(self, tokens, num_samples=1, **kwargs):
        """tokens: list[int] prompt. Returns (results, masks): results is a list of num_samples
        token-id lists (the prompt plus its generated continuation), masks a matching list of
        0/1 lists (0 = forced/tool-injected, 1 = sampled) -- see nanochat.engine.Engine.generate_batch,
        the reference implementation. Only `results` is used by benchcore's generative eval loop."""
        ...

    # OPTIONAL, not declared here (see module docstring): a generator that can decode several
    # DIFFERENT prompts in one batch offers
    #   generate_batch_multi(prompts, num_samples=1, **kwargs)
    # prompts: list[list[int]], one prompt per problem. Returns (results, masks) nested one level
    # deeper than generate_batch: results[p] is the num_samples token-id lists for prompts[p] (each
    # the prompt plus its continuation), masks[p] the matching 0/1 lists; same kwargs as
    # generate_batch. Contract: generate_batch_multi([t], n) is generate_batch(t, n) wrapped in a
    # one-element list. benchcore's generative loop uses it when asked for a batch size above 1 and
    # otherwise -- or when a generator lacks it -- falls back to one generate_batch per problem, so
    # a generator without it keeps working unmodified. It is a separate method rather than a wider
    # generate_batch because a wider argument cannot be feature-detected (getattr finds the method
    # either way) and one method returning flat-or-nested rows would index fine in both shapes.
