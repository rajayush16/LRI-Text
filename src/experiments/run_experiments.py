"""One-command reproduction of the full experiment suite, in order.
Equivalent to running each experiment script manually (see README.md); this
just chains them for convenience. Each sub-experiment is independently
runnable and idempotent (results are appended, checkpoints are per-run-id).
"""

import argparse
import subprocess
import sys

STEPS = [
    ("A: baselines", [sys.executable, "-m", "src.models.baseline"]),
    ("C: low-resource transformer finetune", [sys.executable, "-m", "src.training.trainer", "--experiment", "experiment_C_low_resource"]),
    ("B: full-data transformer finetune", [sys.executable, "-m", "src.training.trainer", "--experiment", "experiment_B_full_finetune"]),
    ("D: back-translation augmentation", [sys.executable, "-m", "src.experiments.augmentation_experiment"]),
    ("E: few-shot", [sys.executable, "-m", "src.experiments.few_shot"]),
    ("F: cross-lingual transfer", [sys.executable, "-m", "src.experiments.cross_lingual"]),
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip", nargs="*", default=[], help="Step name substrings to skip, e.g. --skip D E")
    args = parser.parse_args()

    for name, cmd in STEPS:
        if any(s in name for s in args.skip):
            print(f"[SKIP] {name}")
            continue
        print(f"[RUN] {name}: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
