from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
from sklearn.metrics import average_precision_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer


ROOT = Path(__file__).parent
DATA = Path("/home/dzkhomidov/matching-work/rescue_20260824/macro_balance/inputs/hand_pairs_pd_v3cal.parquet")
HARD = Path("/home/dzkhomidov/matching-work/data/hand_pairs.parquet")
MODEL_NAME = "DeepPavlov/rubert-base-cased"
VARIANTS = {"e2_len224": 224, "e2_len384": 384, "e3_len384": 384}
FOLDS = tuple(f"fold_{i:02d}" for i in range(1, 5))


def tokenize(df, tokenizer, max_len):
    def texts(side):
        out = []
        for name, attrs, category in zip(df[f"name{side}"], df[f"attrs{side}"], df["category"]):
            text = f"{name} | {category}"
            if attrs:
                text += f" | {attrs}"
            out.append(text)
        return out
    left, right = texts(1), texts(2)
    n = df.height
    ids = np.zeros((n, max_len), dtype=np.int32)
    tt = np.zeros((n, max_len), dtype=np.uint8)
    for start in range(0, n, 20_000):
        stop = min(start + 20_000, n)
        batch = tokenizer(left[start:stop], right[start:stop], truncation=True,
                          max_length=max_len, padding="max_length", return_tensors="np")
        ids[start:stop] = batch["input_ids"].astype(np.int32)
        if "token_type_ids" in batch:
            tt[start:stop] = batch["token_type_ids"].astype(np.uint8)
    return ids, tt


def main():
    started = time.time()
    df = pl.read_parquet(DATA)
    hard = pl.read_parquet(HARD).select("id1", "id2", "fold", "category", "target")
    assert df.select("id1", "id2", "fold").equals(hard.select("id1", "id2", "fold"))
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    fold_values = df["fold"].to_numpy()
    metrics = []
    for max_len in sorted(set(VARIANTS.values())):
        ids, tt = tokenize(df, tokenizer, max_len)
        for variant, variant_len in VARIANTS.items():
            if variant_len != max_len:
                continue
            outdir = ROOT / "single_direction_preds" / variant
            outdir.mkdir(parents=True, exist_ok=True)
            for fold in FOLDS:
                rows = np.flatnonzero(fold_values == fold)
                pred_path = outdir / f"{fold}.csv"
                if pred_path.exists():
                    saved = pl.read_csv(pred_path)
                    assert saved.height == len(rows)
                    pred = saved["predict"].to_numpy()
                    model = None
                else:
                    model_dir = ROOT / "checkpoints" / variant / fold
                    model = AutoModelForSequenceClassification.from_pretrained(model_dir).cuda().eval()
                    use_tt = getattr(model.config, "type_vocab_size", 0) > 1
                    pred = np.empty(len(rows), dtype=np.float64)
                    with torch.inference_mode():
                        for start in range(0, len(rows), 512):
                            part = rows[start:start + 512]
                            bi = torch.from_numpy(ids[part].astype(np.int64)).cuda()
                            bt = torch.from_numpy(tt[part].astype(np.int64)).cuda()
                            with torch.autocast("cuda", dtype=torch.bfloat16):
                                logits = model(input_ids=bi,
                                               attention_mask=(bi != tokenizer.pad_token_id).long(),
                                               token_type_ids=bt if use_tt else None).logits.squeeze(-1)
                            pred[start:start + len(part)] = torch.sigmoid(logits.float()).cpu().numpy()
                h = hard.filter(pl.col("fold") == fold).with_columns(pl.Series("predict", pred))
                single_macro = float(np.mean([
                    average_precision_score(g["target"], g["predict"])
                    for _, g in h.group_by("category")]))
                single_pooled = float(average_precision_score(h["target"], h["predict"]))
                two = pl.read_csv(ROOT / "preds" / variant / f"{fold}.csv")
                joined = hard.filter(pl.col("fold") == fold).join(two, on=["id1", "id2"], validate="1:1")
                two_macro = float(np.mean([
                    average_precision_score(g["target"], g["predict"])
                    for _, g in joined.group_by("category")]))
                two_pooled = float(average_precision_score(joined["target"], joined["predict"]))
                metrics.append({"variant": variant, "fold": fold,
                                "single_macro_ap": single_macro,
                                "two_direction_macro_ap": two_macro,
                                "single_minus_two_macro": single_macro - two_macro,
                                "single_pooled_ap": single_pooled,
                                "two_direction_pooled_ap": two_pooled,
                                "single_minus_two_pooled": single_pooled - two_pooled})
                if not pred_path.exists():
                    pl.DataFrame({"id1": h["id1"], "id2": h["id2"], "predict": pred}).write_csv(pred_path)
                del model
                torch.cuda.empty_cache()
    summary = {variant: {key: float(np.mean([row[key] for row in metrics if row["variant"] == variant]))
                         for key in ("single_macro_ap", "two_direction_macro_ap", "single_minus_two_macro",
                                     "single_pooled_ap", "two_direction_pooled_ap", "single_minus_two_pooled")}
               for variant in VARIANTS}
    result = {"rows": metrics, "summary": summary, "runtime_seconds": time.time() - started}
    (ROOT / "single_direction_metrics_4fold.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
