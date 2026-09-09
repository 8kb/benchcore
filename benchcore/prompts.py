"""
CORE prompt rendering: plain-Python string building, replacing the jinja2 templates
nanochat/core_eval.py originally used (dropped to trim benchcore's dependency footprint --
verified byte-identical to the jinja2 output via nanochat's tests/goldens/eval_core_prompts.json,
captured before the move).

Each render_prompts_* function returns the complete prompt string(s) for one example, with
few-shot examples prepended in the same format the item itself uses.
"""
import torch


def render_prompts_mc(item, continuation_delimiter, fewshot_examples=None):
    """Render complete prompts for a multiple choice question -- one prompt per choice."""
    fewshot_examples = fewshot_examples or []
    prefix = "".join(
        f"{ex['query']}{continuation_delimiter}{ex['choices'][ex['gold']]}\n\n"
        for ex in fewshot_examples
    )
    return [f"{prefix}{item['query']}{continuation_delimiter}{choice}" for choice in item['choices']]


def render_prompts_schema(item, continuation_delimiter, fewshot_examples=None):
    """Render complete prompts for a schema question -- one prompt per context option."""
    fewshot_examples = fewshot_examples or []
    prefix = "".join(
        f"{ex['context_options'][ex['gold']]}{continuation_delimiter}{ex['continuation']}\n\n"
        for ex in fewshot_examples
    )
    return [
        f"{prefix}{context_option}{continuation_delimiter}{item['continuation']}"
        for context_option in item['context_options']
    ]


def render_prompts_lm(item, continuation_delimiter, fewshot_examples=None):
    """
    Render complete prompt for a language modeling task. Contexts are stripped (trimmed) before
    joining -- some datasets store them with trailing whitespace, which would otherwise get
    absorbed into the next token in prompt_with. Returns two prompts: without and with the
    continuation.
    """
    fewshot_examples = fewshot_examples or []
    prefix = "".join(
        f"{ex['context'].strip()}{continuation_delimiter}{ex['continuation']}\n\n"
        for ex in fewshot_examples
    )
    base = f"{prefix}{item['context'].strip()}{continuation_delimiter}"
    prompt_with = f"{base}{item['continuation']}"
    # Strip the whole thing (not just the context) so a trailing-whitespace continuation_delimiter
    # (e.g. the default " ") doesn't leave prompt_without with a dangling trailing token boundary
    # that prompt_with's tokenization wouldn't share -- see the docstring above.
    prompt_without = base.strip()
    return [prompt_without, prompt_with]


def find_common_length(token_sequences, direction='left'):
    """
    Find the length of the common prefix or suffix across token sequences.
    direction: 'left' for prefix, 'right' for suffix.
    """
    min_len = min(len(seq) for seq in token_sequences)
    indices = {
        'left': range(min_len),
        'right': range(-1, -min_len - 1, -1),
    }[direction]
    for i, idx in enumerate(indices):
        token = token_sequences[0][idx]
        if not all(seq[idx] == token for seq in token_sequences):
            return i
    return min_len


def stack_sequences(tokens, pad_token_id):
    """Stack up a list of token sequences, pad to longest on the right."""
    bsz, seq_len = len(tokens), max(len(x) for x in tokens)
    input_ids = torch.full((bsz, seq_len), pad_token_id, dtype=torch.long)
    for i, x in enumerate(tokens):
        input_ids[i, :len(x)] = torch.tensor(x, dtype=torch.long)
    return input_ids


def batch_sequences_mc(tokenizer, prompts):
    # In multiple choice, contexts are the same but the continuation is different (common prefix)
    tokens = tokenizer.encode(prompts, prepend=tokenizer.get_bos_token_id())
    answer_start_idx = find_common_length(tokens, direction='left')
    start_indices = [answer_start_idx] * len(prompts)
    end_indices = [len(x) for x in tokens]
    return tokens, start_indices, end_indices


def batch_sequences_schema(tokenizer, prompts):
    # In schema tasks, contexts vary but continuation is the same (common suffix)
    tokens = tokenizer.encode(prompts, prepend=tokenizer.get_bos_token_id())
    suffix_length = find_common_length(tokens, direction='right')
    end_indices = [len(x) for x in tokens]
    start_indices = [ei - suffix_length for ei in end_indices]
    return tokens, start_indices, end_indices


def batch_sequences_lm(tokenizer, prompts):
    # In LM tasks, we have two prompts: without and with continuation
    tokens = tokenizer.encode(prompts, prepend=tokenizer.get_bos_token_id())
    tokens_without, tokens_with = tokens
    start_idx, end_idx = len(tokens_without), len(tokens_with)
    assert start_idx < end_idx, "prompt without is supposed to be a prefix of prompt with"
    assert tokens_without == tokens_with[:start_idx], "prompt without is supposed to be a prefix of prompt with"
    # we only need the with continuation prompt in the LM task, i.e. batch size of 1
    return [tokens_with], [start_idx], [end_idx]
