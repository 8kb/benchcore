"""
Test benchcore.suite: the centering formula, and load_core_suite against a synthetic bundle
written to disk (no network, no real eval_bundle.zip).

python -m pytest benchcore/tests/test_suite.py -v
"""
import csv
import json
import os

import yaml

from benchcore.suite import center, load_core_suite


def test_center_zero_at_baseline_one_at_perfect():
    # accuracy exactly at the random baseline centers to 0
    assert center(0.25, random_baseline=25.0) == 0.0
    # perfect accuracy centers to 1, regardless of baseline
    assert center(1.0, random_baseline=25.0) == 1.0
    # a hand-computed midpoint
    assert abs(center(0.625, random_baseline=25.0) - 0.5) < 1e-9


def _write_synthetic_bundle(root):
    bundle_dir = os.path.join(root, "eval_bundle")
    data_dir = os.path.join(bundle_dir, "eval_data")
    os.makedirs(data_dir, exist_ok=True)

    core_yaml = {
        "icl_tasks": [
            {
                "label": "toy_mc",
                "icl_task_type": "multiple_choice",
                "dataset_uri": "toy_mc.jsonl",
                "num_fewshot": [0],
                "continuation_delimiter": " ",
            },
        ],
    }
    with open(os.path.join(bundle_dir, "core.yaml"), "w") as f:
        yaml.safe_dump(core_yaml, f)

    with open(os.path.join(bundle_dir, "eval_meta_data.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Eval Task", "Random baseline"])
        writer.writeheader()
        writer.writerow({"Eval Task": "toy_mc", "Random baseline": "50.0"})

    examples = [
        {"query": f"item {i}", "choices": ["yes", "no"], "gold": i % 2}
        for i in range(10)
    ]
    with open(os.path.join(data_dir, "toy_mc.jsonl"), "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")


def test_load_core_suite_from_synthetic_bundle(tmp_path):
    _write_synthetic_bundle(str(tmp_path))
    suite = load_core_suite(str(tmp_path))
    assert len(suite.tasks) == 1
    task = suite.tasks[0]
    assert task.label == "toy_mc"
    assert task.task_type == "multiple_choice"
    assert task.num_fewshot == 0
    assert len(task.data) == 10
    assert suite.random_baselines["toy_mc"] == 50.0


def test_load_core_suite_max_per_task_subsamples_deterministically(tmp_path):
    _write_synthetic_bundle(str(tmp_path))
    suite_a = load_core_suite(str(tmp_path), max_per_task=3)
    suite_b = load_core_suite(str(tmp_path), max_per_task=3)
    assert len(suite_a.tasks[0].data) == 3
    # the same seed subsamples the same examples every time
    assert suite_a.tasks[0].data == suite_b.tasks[0].data


def test_load_core_suite_does_not_redownload_when_bundle_exists(tmp_path, monkeypatch):
    _write_synthetic_bundle(str(tmp_path))

    def _fail(*args, **kwargs):
        raise AssertionError("should not attempt to download when the bundle already exists")

    monkeypatch.setattr("benchcore.suite.download_file", _fail)
    load_core_suite(str(tmp_path)) # must not raise
