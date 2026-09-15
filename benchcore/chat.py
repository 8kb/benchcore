"""
Chat-style task evaluation: a generative loop (sample from the model, check the completion) and a
categorical loop (no sampling -- read off logits over a fixed set of answer letters), plus the
ChatCORE metric (mean centered accuracy across a fixed task suite, 0 = random baseline, 1 =
perfect). Ported from scripts/chat_eval.py.

No ambient distributed state: both loops take explicit rank/world_size/device and return each
rank's partial (num_passed, total) rather than calling dist.all_reduce themselves -- mirrors
benchcore.core.evaluate_task and datacore.DataManager.batches(). A caller under torchrun reduces
across ranks itself, or uses BenchManager.chat()/chat_suite() which does this for it.
"""
import torch

# The standard five-task ChatCORE suite and each task's random-guess baseline accuracy.
ALL_CHAT_TASKS = ('ARC-Easy', 'ARC-Challenge', 'MMLU', 'GSM8K', 'HumanEval')
CATEGORICAL_CHAT_TASKS = ('ARC-Easy', 'ARC-Challenge', 'MMLU')
CHAT_BASELINE_ACCURACIES = {
    'ARC-Easy': 0.25, # multiple choice 1 of 4 => 25%
    'ARC-Challenge': 0.25, # multiple choice 1 of 4 => 25%
    'MMLU': 0.25, # multiple choice 1 of 4 => 25%
    'GSM8K': 0.0, # open-ended => 0%
    'HumanEval': 0.0, # open-ended => 0%
}


def run_generative_eval(task_object, tokenizer, generator, num_samples, max_new_tokens, temperature, top_k,
                         max_problems=None, *, rank=0, world_size=1):
    """We go one problem at a time, sample, evaluate. Returns this rank's (num_passed, total)."""
    num_problems = len(task_object) if max_problems is None else min(len(task_object), max_problems)
    num_passed, total = 0, 0
    for i in range(rank, num_problems, world_size):
        conversation = task_object[i]
        encoded_prompt = tokenizer.render_for_completion(conversation)
        results, _ = generator.generate_batch(
            encoded_prompt,
            num_samples=num_samples,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )
        prefix_length = len(encoded_prompt)
        completions = [tokenizer.decode(result_tokens[prefix_length:]) for result_tokens in results]
        outcomes = [task_object.evaluate(conversation, completion) for completion in completions]
        passed = any(outcomes)
        total += 1
        num_passed += int(passed)
    return num_passed, total


def run_categorical_eval(task_object, tokenizer, model, batch_size, max_problems=None, *, device=None, rank=0, world_size=1):
    """
    A lot easier because we don't have to sample: batches of independent problems at a time,
    checking the logits for correct answer choices. Returns this rank's (num_passed, total).
    """
    device = device if device is not None else getattr(model, "get_device", lambda: torch.device("cpu"))()
    bos = tokenizer.get_bos_token_id() # use BOS as pad token is ok, these positions are ignored

    num_problems = len(task_object) if max_problems is None else min(len(task_object), max_problems)
    ceil_div = lambda x, y: -(-x // y)
    num_batches = ceil_div(num_problems, batch_size)

    letter_to_id_cache = {} # many letters will repeat often, let's save the tokenizer some work
    num_passed, total = 0, 0
    for i in range(rank, num_batches, world_size):
        i0, i1 = i * batch_size, min((i + 1) * batch_size, num_problems)

        conversations = [task_object[ii] for ii in range(i0, i1)]
        prompt_ids = [tokenizer.render_for_completion(conversation) for conversation in conversations]
        max_length = max(len(ids) for ids in prompt_ids)
        answer_time_positions = [len(ids) - 1 for ids in prompt_ids]
        padded_prompt_ids = [ids + [bos] * (max_length - len(ids)) for ids in prompt_ids]
        prompt_ids = torch.tensor(padded_prompt_ids, dtype=torch.long, device=device)

        with torch.no_grad():
            logits = model(prompt_ids) # (B, T, V)

        for idx, conversation in enumerate(conversations):
            letters = conversation['letters']
            letter_ids = []
            for letter in letters:
                if letter not in letter_to_id_cache:
                    encoded_letter = tokenizer.encode(letter)
                    assert len(encoded_letter) == 1, "Each letter must be a single token"
                    letter_to_id_cache[letter] = encoded_letter[0]
                letter_ids.append(letter_to_id_cache[letter])
            answer_pos = answer_time_positions[idx]
            focus_logits = logits[idx, answer_pos, letter_ids]
            argmax_letter_id = focus_logits.argmax(dim=-1).item()
            predicted_letter = letters[argmax_letter_id]
            outcome = task_object.evaluate(conversation, predicted_letter)
            num_passed += int(outcome)
            total += 1

    return num_passed, total


def chatcore_metric(task_accuracies, tasks=ALL_CHAT_TASKS, baselines=CHAT_BASELINE_ACCURACIES):
    """Mean centered accuracy over `tasks`: 0 at each task's random baseline, 1 at perfect."""
    centered = [
        (task_accuracies[t] - baselines.get(t, 0.0)) / (1.0 - baselines.get(t, 0.0))
        for t in tasks
    ]
    return sum(centered) / len(centered)


def build_chat_tasks(names=None, *, cache_dir):
    """{task_name: Task} for `names` (default: every name in ALL_CHAT_TASKS), each built against
    its test split at `cache_dir`. Moved here from three identical copies (nanochat's
    scripts/chat_eval.py, scripts/chat_sft.py's own in-training ChatCORE check, tinylab's
    tinylab/ops/bench.py) -- benchcore already owns both the names and the task classes, so this
    was benchcore's own mapping duplicated at every call site. Raises ValueError naming the
    unknown entries if `names` has any name outside ALL_CHAT_TASKS."""
    from benchcore.tasks import ARC, GSM8K, MMLU, HumanEval
    builders = {
        'ARC-Easy': lambda: ARC(subset="ARC-Easy", split="test", cache_dir=cache_dir),
        'ARC-Challenge': lambda: ARC(subset="ARC-Challenge", split="test", cache_dir=cache_dir),
        'MMLU': lambda: MMLU(subset="all", split="test", cache_dir=cache_dir),
        'GSM8K': lambda: GSM8K(subset="main", split="test", cache_dir=cache_dir),
        'HumanEval': lambda: HumanEval(cache_dir=cache_dir),
    }
    names = list(ALL_CHAT_TASKS) if names is None else list(names)
    unknown = set(names) - set(builders)
    if unknown:
        raise ValueError(f"Unknown chat task(s) {sorted(unknown)} (available: {sorted(builders)})")
    return {name: builders[name]() for name in names}
