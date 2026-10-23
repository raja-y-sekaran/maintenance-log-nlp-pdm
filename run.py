"""Run the full pipeline on the synthetic maintenance log and draw the figures."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from pdm_nlp import figures, pipeline  # noqa: E402

if __name__ == "__main__":
    m, test, scores, failures, episodes, chosen = pipeline.run()
    figures.draw(test, scores, failures, episodes, chosen)
    print(json.dumps(m, indent=2, default=str))
