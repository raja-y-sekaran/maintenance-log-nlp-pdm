"""Run the project end to end (course layout: data -> features -> models -> evaluation -> saved model).

    python main.py                       # all stages on the data in data/raw/ (about 30 s)
    python main.py --regenerate --seed 7 # make a new synthetic plant first
    python main.py --robustness          # repeat on five synthetic plants, then restore seed 42
"""
import argparse
import json
import subprocess
import sys

from src import pipeline
from src.data import generate_synthetic
from src.models import train
from src.utils import plots
from src.utils.paths import REPORTS


def run_all(quiet=False):
    m, test, scores, failures, episodes, chosen = pipeline.run()  # stages 2-7
    plots.draw(test, scores, failures, episodes, chosen)
    path = train.train_and_save()                                  # stage 6: model for the API
    if not quiet:
        print(json.dumps({k: m[k] for k in ("chosen_model", "test_pr_auc", "test_precision_at_k")}, indent=2))
        print(f"model saved to {path.relative_to(path.parents[1])}")
    return m


def robustness(seeds=(1, 2, 3, 4, 42)):
    print("| Seed | Base rate | Keyword PR-AUC | History PR-AUC | Text+LR PR-AUC | Text+LightGBM PR-AUC | Chosen model: breakdowns with text warning caught | Median lead (days) |")
    print("|---|---|---|---|---|---|---|---|")
    for s in seeds:
        subprocess.run([sys.executable, "-m", "src.data.generate_synthetic", "--seed", str(s)], check=True, capture_output=True)
        subprocess.run([sys.executable, "main.py", "--quiet"], check=True, capture_output=True)
        m = json.loads((REPORTS / "metrics.json").read_text())
        a, c = m["test_pr_auc"], m["test_breakdowns_caught"][m["chosen_model"]]
        print(f"| {s} | {m['test_base_rate']:.3f} | {a['Keyword count']:.2f} | {a['History only (no text)']:.2f} | "
              f"{a['Text + history, logistic']:.2f} | {a['Text + history, LightGBM']:.2f} | "
              f"{c['with_warning_in_text']} | {c['median_lead_days']:.0f} |")
    subprocess.run([sys.executable, "-m", "src.data.generate_synthetic", "--seed", "42"], check=True, capture_output=True)
    subprocess.run([sys.executable, "main.py", "--quiet"], check=True, capture_output=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--regenerate", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--robustness", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    if a.robustness:
        robustness()
    else:
        if a.regenerate:
            sys.argv = ["generate_synthetic", "--seed", str(a.seed)]
            generate_synthetic.main()
        run_all(a.quiet)
