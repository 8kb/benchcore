"""
The CORE benchmark bundle: fetches/unzips DCLM's eval_bundle.zip, parses its core.yaml (task
list) and eval_meta_data.csv (random baselines), and holds the per-example data for each task.
Ported from scripts/base_eval.py's evaluate_core() (bundle handling) plus the centering math that
lived alongside it.

Requires the `bundle` extra (pyyaml) -- only this module needs it.
"""
import csv
import json
import os
import random
import shutil
import tempfile
import zipfile
from dataclasses import dataclass, field

from datacore.download import download_file

EVAL_BUNDLE_URL = "https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip"


@dataclass
class CoreTask:
    label: str
    task_type: str # 'multiple_choice' | 'schema' | 'language_modeling'
    num_fewshot: int
    continuation_delimiter: str
    data: list # list of example dicts, already loaded from the bundle's jsonl

    @property
    def task_meta(self):
        return {
            'task_type': self.task_type,
            'num_fewshot': self.num_fewshot,
            'continuation_delimiter': self.continuation_delimiter,
        }


@dataclass
class CoreSuite:
    tasks: list # list[CoreTask]
    random_baselines: dict # label -> float (0-100 scale, as stored in eval_meta_data.csv)


def _place_eval_bundle(zip_path, cache_dir):
    """Unzip eval_bundle.zip into cache_dir/eval_bundle."""
    eval_bundle_dir = os.path.join(cache_dir, "eval_bundle")
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(tmpdir)
        extracted_bundle_dir = os.path.join(tmpdir, "eval_bundle")
        shutil.move(extracted_bundle_dir, eval_bundle_dir)


def load_core_suite(cache_dir, *, max_per_task=None, seed=1337):
    """
    Downloads (if needed) and loads the CORE bundle from cache_dir/eval_bundle, returning a
    CoreSuite with every task's data pre-loaded and (optionally) subsampled deterministically to
    max_per_task examples -- matching scripts/base_eval.py's original shuffle-then-slice behavior.
    """
    eval_bundle_dir = os.path.join(cache_dir, "eval_bundle")
    if not os.path.exists(eval_bundle_dir):
        os.makedirs(cache_dir, exist_ok=True)
        zip_path = os.path.join(cache_dir, "eval_bundle.zip")
        ok = download_file(EVAL_BUNDLE_URL, zip_path)
        if not ok:
            raise RuntimeError(f"Failed to download CORE eval bundle from {EVAL_BUNDLE_URL}")
        _place_eval_bundle(zip_path, cache_dir)

    import yaml # optional [bundle] dependency -- only this function needs it

    config_path = os.path.join(eval_bundle_dir, "core.yaml")
    data_base_path = os.path.join(eval_bundle_dir, "eval_data")
    eval_meta_data = os.path.join(eval_bundle_dir, "eval_meta_data.csv")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    random_baselines = {}
    with open(eval_meta_data, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            random_baselines[row['Eval Task']] = float(row['Random baseline'])

    tasks = []
    for task_spec in config['icl_tasks']:
        label = task_spec['label']
        data_path = os.path.join(data_base_path, task_spec['dataset_uri'])
        with open(data_path, 'r', encoding='utf-8') as f:
            data = [json.loads(line.strip()) for line in f]
        shuffle_rng = random.Random(seed)
        shuffle_rng.shuffle(data)
        if max_per_task is not None and max_per_task > 0:
            data = data[:max_per_task]
        tasks.append(CoreTask(
            label=label,
            task_type=task_spec['icl_task_type'],
            num_fewshot=task_spec['num_fewshot'][0],
            continuation_delimiter=task_spec.get('continuation_delimiter', ' '),
            data=data,
        ))

    return CoreSuite(tasks=tasks, random_baselines=random_baselines)


def center(accuracy, random_baseline):
    """Centers a 0-1 accuracy against a 0-100-scale random baseline: 0 at random, 1 at perfect."""
    return (accuracy - 0.01 * random_baseline) / (1.0 - 0.01 * random_baseline)
