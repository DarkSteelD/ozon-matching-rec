import argparse
import json
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("gate_json")
parser.add_argument("output_json")
args = parser.parse_args()
scores = {row["variant"]: row for row in json.loads(Path(args.gate_json).read_text())["scores"]}
baseline = scores["bce"]
base_folds = {row["fold"]: row["macro_category_ap"] for row in baseline["folds"]}
decisions = []
for name, candidate in scores.items():
    if name == "bce":
        continue
    deltas = {row["fold"]: row["macro_category_ap"] - base_folds[row["fold"]]
              for row in candidate["folds"]}
    pooled_delta = candidate["macro_category_ap"] - baseline["macro_category_ap"]
    decisions.append({
        "variant": name,
        "fold_deltas": deltas,
        "pooled_macro_category_ap_delta": pooled_delta,
        "gate": "GO" if all(delta > 0 for delta in deltas.values()) and pooled_delta > 0.001 else "NO_GO",
    })
result = {"baseline_macro_category_ap": baseline["macro_category_ap"], "decisions": decisions}
Path(args.output_json).write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
