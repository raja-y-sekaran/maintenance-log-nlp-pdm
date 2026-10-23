"""Re-run everything on several synthetic plants (seeds) so the results are not one lucky draw.
Restores the default seed-42 data and outputs at the end.   python scripts/robustness.py"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEEDS = [1, 2, 3, 4, 42]


def sh(*args):
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True, capture_output=True, text=True)


print("| Seed | Base rate | Keyword PR-AUC | History PR-AUC | Text+LR PR-AUC | Text+LightGBM PR-AUC | Chosen model: breakdowns with text warning caught | Median lead (days) |")
print("|---|---|---|---|---|---|---|---|")
for s in SEEDS:
    sh("data/generate_synthetic.py", "--seed", str(s))
    sh("run.py")
    m = json.loads((ROOT / "outputs" / "metrics.json").read_text())
    a = m["test_pr_auc"]
    c = m["test_breakdowns_caught"][m["chosen_model"]]
    print(f"| {s} | {m['test_base_rate']:.3f} | {a['Keyword count']:.2f} | {a['History only (no text)']:.2f} | "
          f"{a['Text + history, logistic']:.2f} | {a['Text + history, LightGBM']:.2f} | "
          f"{c['with_warning_in_text']} | {c['median_lead_days']:.0f} |")
sh("data/generate_synthetic.py", "--seed", "42")
sh("run.py")
