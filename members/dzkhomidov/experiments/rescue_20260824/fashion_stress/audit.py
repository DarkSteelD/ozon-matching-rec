#!/usr/bin/env python3
"""Deterministic CPU-only stress audit for matching fashion OOF predictions."""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


ROOT = Path("/home/dzkhomidov/ozon-hack/repos/ozon-matching-rec")
WORK = Path("/home/dzkhomidov/matching-work")
CATS = ["Обувь", "Одежда", "Галантерея и аксессуары", "Ювелирные изделия"]
BASE = "final_stack_all"
ALTS = ["zs_llm_blend", "knrm_name_v2", "ce_priodistill", "ce_fashion_specialist"]
INDEPENDENT = {"zs_llm_blend", "knrm_name_v2"}
PUBLIC_PRIOR = {
    "Обувь": 0.045808781600456185,
    "Одежда": 0.057251184834123225,
    "Галантерея и аксессуары": 0.043035306516774986,
    "Ювелирные изделия": 0.015708741452596563,
}
PUBLIC_BASE_AP = {
    "Обувь": 0.09652270022990363,
    "Одежда": 0.11084216129694444,
    "Галантерея и аксессуары": 0.24218789350808034,
    "Ювелирные изделия": 0.2865652517631082,
}
BRAND_KEYS = ("бренд", "brand", "марка")
MODEL_KEYS = ("модель", "model", "линейка", "коллекция")
ARTICLE_KEYS = ("артикул", "партномер", "sku", "код товара", "код производителя")
PACK_KEYS = ("упаков", "транспорт", "габарит", "длина", "ширина", "высота")
SCORE_EDGES = [0, .01, .05, .2, .5, .8, .95, .99, 1.0000001]
SCORE_LABELS = ["score_00_01", "score_01_05", "score_05_20", "score_20_50",
                "score_50_80", "score_80_95", "score_95_99", "score_99_100"]
TOKEN_RE = re.compile(r"[a-zа-яё0-9]+", re.I)
NUM_RE = re.compile(r"(?<![a-zа-яё])\d+(?:[.,]\d+)?(?![a-zа-яё])", re.I)


def norm(s: object) -> str:
    return " ".join(TOKEN_RE.findall(str(s).casefold()))


def values(attrs: dict, needles: tuple[str, ...]) -> set[str]:
    out = set()
    for key, value in attrs.items():
        if any(x in str(key).casefold() for x in needles):
            seq = value if isinstance(value, list) else re.split(r"[;|,/]+", str(value))
            out.update(v for x in seq if (v := norm(x)))
    return out


def relation(a: set[str], b: set[str], prefix: str) -> str:
    if not a and not b:
        return f"{prefix}_both_missing"
    if not a or not b:
        return f"{prefix}_one_missing"
    return f"{prefix}_agree" if a & b else f"{prefix}_conflict"


def number_tokens(s: str) -> set[str]:
    return {x.replace(",", ".").lstrip("0") or "0" for x in NUM_RE.findall(s.casefold())}


def size_values(attrs: dict) -> dict[str, set[str]]:
    out: dict[str, set[str]] = defaultdict(set)
    for key, value in attrs.items():
        k = str(key).casefold()
        if "размер" not in k or any(x in k for x in PACK_KEYS):
            continue
        system = "ru" if "россий" in k or "ru" in k else "maker"
        text = str(value).casefold()
        nums = number_tokens(text)
        letters = {x.upper() for x in re.findall(r"(?<![a-z])(?:xxs|xs|s|m|l|xl|xxl|xxxl)(?![a-z])", text)}
        if letters and not nums:
            system = "letter"
            out[system].update(letters)
        else:
            out[system].update(nums)
    return out


def size_relation(a: dict[str, set[str]], b: dict[str, set[str]]) -> str:
    common = set(a) & set(b)
    if not common:
        return "size_not_comparable"
    return "size_comparable_agree" if any(a[k] & b[k] for k in common) else "size_comparable_mismatch"


def parse_attrs(s: object) -> dict:
    try:
        x = json.loads(s) if isinstance(s, str) and s else {}
        return x if isinstance(x, dict) else {}
    except json.JSONDecodeError:
        return {}


def ap(y: np.ndarray, score: np.ndarray, weight: np.ndarray | None = None) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, score, sample_weight=weight))


def rank_parent(df: pd.DataFrame, col: str) -> pd.Series:
    return df.groupby(["category", "fold"], observed=True)[col].rank(pct=True, method="average")


def weighted_ap_plan(y: np.ndarray, score: np.ndarray, group_code: np.ndarray) -> tuple:
    order = np.argsort(-score, kind="stable")
    sorted_score = score[order]
    starts = np.r_[0, np.flatnonzero(sorted_score[1:] != sorted_score[:-1]) + 1]
    return y[order], group_code[order], starts


def weighted_ap_from_counts(plan: tuple, group_counts: np.ndarray) -> float:
    y, group_code, starts = plan
    weight = group_counts[group_code]
    tp = np.add.reduceat(weight * y, starts)
    total = np.add.reduceat(weight, starts)
    cum_tp, cum_total = np.cumsum(tp), np.cumsum(total)
    if not len(cum_tp) or cum_tp[-1] == 0 or cum_tp[-1] == cum_total[-1]:
        return float("nan")
    precision = np.divide(cum_tp, cum_total, out=np.zeros_like(cum_tp, dtype=float), where=cum_total > 0)
    return float(np.sum(precision * (tp / cum_tp[-1])))


def add_components(df: pd.DataFrame) -> pd.Series:
    parent: dict[int, int] = {}

    def find(x: int) -> int:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        a, b = find(a), find(b)
        if a != b:
            parent[max(a, b)] = min(a, b)

    for a, b in zip(df.id1, df.id2):
        union(int(a), int(b))
    return pd.Series([find(int(a)) for a in df.id1], index=df.index, dtype="int64")


def build_rows() -> tuple[pd.DataFrame, pd.DataFrame]:
    pred = pd.read_parquet(ROOT / "members/dzkhomidov/preds/all_model_predictions_oof.parquet")
    pred = pred[pred.category.isin(CATS)].copy()
    pairs = pd.read_parquet(WORK / "data/hand_pairs.parquet",
                            columns=["fold", "id1", "id2", "name1", "name2"])
    df = pred.merge(pairs, on=["fold", "id1", "id2"], how="left", validate="one_to_one")
    ids = pd.unique(df[["id1", "id2"]].values.ravel())
    items = pd.read_parquet(ROOT / "data/raw/items_human.parquet",
                            columns=["id", "attributes"])
    items = items[items.id.isin(ids)].set_index("id").attributes.map(parse_attrs)
    assert df.name1.notna().all() and set(ids) <= set(items.index), "pair/item join incomplete"

    records = []
    for row in df.itertuples(index=False):
        a, b = items.at[row.id1], items.at[row.id2]
        n1, n2 = norm(row.name1), norm(row.name2)
        ratio = SequenceMatcher(None, n1, n2, autojunk=False).ratio()
        nums1, nums2 = number_tokens(row.name1), number_tokens(row.name2)
        maxlen = max(len(str(row.name1)) + len(json.dumps(a, ensure_ascii=False)),
                     len(str(row.name2)) + len(json.dumps(b, ensure_ascii=False)))
        minlen = min(len(str(row.name1)) + len(json.dumps(a, ensure_ascii=False)),
                     len(str(row.name2)) + len(json.dumps(b, ensure_ascii=False)))
        records.append({
            "name_mechanism": "name_exact" if n1 == n2 else "name_near" if ratio >= .85 else "name_other",
            "name_ratio": ratio,
            "attrs_mechanism": "attrs_both_missing" if not a and not b else "attrs_one_missing" if not a or not b else "attrs_present",
            "brand_mechanism": relation(values(a, BRAND_KEYS), values(b, BRAND_KEYS), "brand"),
            "model_mechanism": relation(values(a, MODEL_KEYS), values(b, MODEL_KEYS), "model"),
            "article_mechanism": relation(values(a, ARTICLE_KEYS), values(b, ARTICLE_KEYS), "article"),
            "size_mechanism": size_relation(size_values(a), size_values(b)),
            "numeric_mechanism": "numeric_overlap" if nums1 & nums2 else "numeric_conflict" if nums1 and nums2 else "numeric_missing",
            "length_mechanism": "text_short_both" if maxlen <= 160 else "text_medium" if maxlen <= 800 else "text_long" if maxlen <= 1600 else "text_very_long",
            "text_asymmetric": maxlen / max(minlen, 1) >= 3,
            "max_text_chars": maxlen,
        })
    feats = pd.DataFrame(records, index=df.index)
    df = pd.concat([df, feats], axis=1)
    df["component_id"] = add_components(df)
    df["score_band"] = pd.cut(df[BASE], SCORE_EDGES, labels=SCORE_LABELS,
                              right=False, include_lowest=True).astype(str)
    df["baseline_rank"] = rank_parent(df, BASE)
    for alt in ALTS:
        df[f"{alt}_rank"] = rank_parent(df, alt)
        df[f"blend_{alt}"] = .9 * df.baseline_rank + .1 * df[f"{alt}_rank"]

    audit = pd.read_json(ROOT / "label_audit.jsonl", lines=True)
    audit = audit[audit.category.isin(CATS)].sort_values("at").drop_duplicates(["id1", "id2"], keep="last")
    df = df.merge(audit[["id1", "id2", "audited_label", "original_target", "auditor", "at"]],
                  on=["id1", "id2"], how="left", validate="one_to_one")
    assert len(df) == len(pred) and set(df.fold.unique()) == {f"fold_0{i}" for i in range(1, 5)}
    return df, audit


def all_slices(df: pd.DataFrame) -> list[tuple[str, pd.Series]]:
    out = [("all", pd.Series(True, index=df.index))]
    for col in ["name_mechanism", "attrs_mechanism", "brand_mechanism", "model_mechanism",
                "article_mechanism", "size_mechanism", "numeric_mechanism",
                "length_mechanism", "score_band"]:
        out.extend((value, df[col].eq(value)) for value in sorted(df[col].dropna().unique()))
    out.append(("text_asymmetric", df.text_asymmetric))
    return out


def metrics(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    slice_rows, blend_rows = [], []
    for cat in CATS:
        for fold in sorted(df.fold.unique()):
            parent = df[(df.category == cat) & (df.fold == fold)]
            parent_ap = ap(parent.target.values, parent[BASE].values)
            for name, mask in all_slices(df):
                sub = df[mask & (df.category == cat) & (df.fold == fold)]
                if not len(sub):
                    continue
                y = sub.target.values
                base_ap = ap(y, sub[BASE].values)
                slice_rows.append({"category": cat, "fold": fold, "slice": name,
                                   "n": len(sub), "positives": int(y.sum()), "negatives": int((1-y).sum()),
                                   "coverage": len(sub) / len(parent), "prevalence": y.mean(),
                                   "baseline_ap": base_ap, "parent_baseline_ap": parent_ap,
                                   "collapse_delta": base_ap - parent_ap,
                                   "groups": sub.component_id.nunique()})
                for alt in ALTS:
                    valid = sub[alt].notna()
                    q = sub[valid]
                    yy = q.target.values
                    base_common = ap(yy, q[BASE].values)
                    alt_ap = ap(yy, q[alt].values)
                    blend_ap = ap(yy, q[f"blend_{alt}"].values)
                    blend_rows.append({"category": cat, "fold": fold, "slice": name, "alternative": alt,
                                       "channel_type": "independent" if alt in INDEPENDENT else "diagnostic_non_independent",
                                       "n": len(q), "positives": int(yy.sum()) if len(q) else 0,
                                       "baseline_ap": base_common, "alternative_ap": alt_ap,
                                       "blend_ap": blend_ap, "blend_delta": blend_ap-base_common,
                                       "status": "checked" if len(np.unique(yy)) == 2 else "undefined_one_class"})
    return pd.DataFrame(slice_rows), pd.DataFrame(blend_rows)


def bootstrap(df: pd.DataFrame, slices: pd.DataFrame, blends: pd.DataFrame,
              n_boot: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    rows, candidate_rows = [], []
    # Score-band bootstrap is intentionally excluded: conditioning on baseline score
    # makes marginal-channel interpretation collider-prone; fold tables retain it.
    mechanisms = [x for x, _ in all_slices(df) if not x.startswith("score_")]
    for cat in CATS:
        cdf = df[df.category == cat]
        for name, mask in all_slices(df):
            if name not in mechanisms:
                continue
            for alt in sorted(INDEPENDENT):
                sf = slices[(slices.category == cat) & (slices.slice == name)]
                bf = blends[(blends.category == cat) & (blends.slice == name) &
                            (blends.alternative == alt) & (blends.status == "checked")]
                supported = (len(sf) == 4 and len(bf) == 4 and (sf.n >= 100).all() and
                             (sf.positives > 0).all() and (sf.negatives > 0).all())
                consistent = supported and (bf.blend_delta > 0).all()
                collapsed = supported and sf.collapse_delta.median() <= -.05
                selected = name == "all" or (consistent and collapsed)
                candidate_rows.append({"category": cat, "slice": name, "alternative": alt,
                                       "four_fold_supported": supported,
                                       "positive_delta_all_folds": consistent,
                                       "median_collapse_le_m05": collapsed,
                                       "selected_for_bootstrap": selected})
                if not selected:
                    continue
                sub = cdf[mask.loc[cdf.index]]
                good = sub[alt].notna()
                q = sub[good]
                if len(q) < 200 or q.target.nunique() < 2:
                    continue
                group_code, groups = pd.factorize(q.component_id, sort=True)
                yv = q.target.to_numpy()
                basev = q[BASE].to_numpy()
                blendv = q[f"blend_{alt}"].to_numpy()
                base_plan = weighted_ap_plan(yv, basev, group_code)
                blend_plan = weighted_ap_plan(yv, blendv, group_code)
                deltas = []
                for _ in range(n_boot):
                    counts = np.bincount(rng.integers(0, len(groups), len(groups)), minlength=len(groups))
                    base_ap = weighted_ap_from_counts(base_plan, counts)
                    blend_ap = weighted_ap_from_counts(blend_plan, counts)
                    if np.isfinite(base_ap) and np.isfinite(blend_ap):
                        deltas.append(blend_ap - base_ap)
                if deltas:
                    lo, med, hi = np.quantile(deltas, [.025, .5, .975])
                    rows.append({"category": cat, "slice": name, "alternative": alt,
                                 "n": len(q), "groups": len(groups), "bootstrap_reps": len(deltas),
                                 "delta_low": lo, "delta_median": med, "delta_high": hi,
                                 "p_delta_gt_0": np.mean(np.asarray(deltas) > 0)})
    return pd.DataFrame(rows), pd.DataFrame(candidate_rows)


def prevalence_shift(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cat in CATS:
        q = df[df.category == cat]
        local = q.target.mean()
        prior = PUBLIC_PRIOR[cat]
        w = np.where(q.target.values == 1, prior/local, (1-prior)/(1-local))
        rows.append({"category": cat, "n": len(q), "local_prior": local, "public_prior": prior,
                     "local_baseline_ap": ap(q.target.values, q[BASE].values),
                     "prevalence_shift_weighted_ap": ap(q.target.values, q[BASE].values, w),
                     "public_baseline_ap": PUBLIC_BASE_AP[cat]})
    out = pd.DataFrame(rows)
    out["public_minus_shift_expectation"] = out.public_baseline_ap - out.prevalence_shift_weighted_ap
    return out


def label_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cat in CATS:
        all_q = df[(df.category == cat) & df.audited_label.notna()].copy()
        q = all_q[all_q.audited_label.isin([0, 1])]
        if not len(all_q):
            rows.append({"category": cat, "n_audited": 0, "status": "unchecked_no_audits"})
            continue
        rows.append({"category": cat, "n_audited": len(all_q), "n_decisive_0_1": len(q),
                     "n_unsure_minus1": int((all_q.audited_label == -1).sum()),
                     "audit_flips": int((q.target != q.audited_label).sum()),
                     "audited_positive": int(q.audited_label.sum()), "audited_negative": int((1-q.audited_label).sum()),
                     "baseline_ap_original_on_selected": ap(q.target.values, q[BASE].values),
                     "baseline_ap_audited_on_selected": ap(q.audited_label.astype(int).values, q[BASE].values),
                     "status": "checked_selection_biased"})
    return pd.DataFrame(rows)


def controls(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for cat in CATS:
        for fold in sorted(df.fold.unique()):
            q = df[(df.category == cat) & (df.fold == fold)]
            y = q.target.values
            base_ap = ap(y, q[BASE].values)
            rows.append({"category": cat, "fold": fold, "variant": BASE,
                         "metric": base_ap, "delta_vs_baseline": 0, "status": "checked"})
            rows.append({"category": cat, "fold": fold, "variant": "positive_control_target",
                         "metric": ap(y, y), "delta_vs_baseline": ap(y, y)-base_ap, "status": "checked"})
            for alt in sorted(INDEPENDENT):
                vals = []
                ar = q[f"{alt}_rank"].values.copy()
                for _ in range(20):
                    rng.shuffle(ar)
                    vals.append(ap(y, .9*q.baseline_rank.values + .1*ar) - base_ap)
                rows.append({"category": cat, "fold": fold, "variant": f"permuted_{alt}",
                             "metric": base_ap + np.mean(vals), "delta_vs_baseline": np.mean(vals),
                             "delta_std_20_permutations": np.std(vals, ddof=1), "status": "checked"})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260824)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=args.output / "run.log", filemode="w", level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", force=True)
    logging.info("start output=%s bootstrap=%d seed=%d", args.output, args.bootstrap, args.seed)

    df, raw_audit = build_rows()
    logging.info("rows built n=%d", len(df))
    slices, blends = metrics(df)
    logging.info("fold metrics built slices=%d blends=%d", len(slices), len(blends))
    boots, candidates = bootstrap(df, slices, blends, args.bootstrap, args.seed)
    logging.info("bootstrap complete selected=%d rows=%d",
                 int(candidates.selected_for_bootstrap.sum()), len(boots))
    prev = prevalence_shift(df)
    labels = label_metrics(df)
    ctrl = controls(df, args.seed)

    keep = ["fold", "id1", "id2", "target", "category", BASE, *ALTS, "component_id",
            "name_mechanism", "name_ratio", "attrs_mechanism", "brand_mechanism",
            "model_mechanism", "article_mechanism", "size_mechanism", "numeric_mechanism",
            "length_mechanism", "text_asymmetric", "max_text_chars", "score_band",
            "audited_label", "original_target"]
    df[keep].to_parquet(args.output / "row_features.parquet", index=False)
    for name, table in [("slice_metrics.csv", slices), ("blend_metrics.csv", blends),
                        ("bootstrap_metrics.csv", boots), ("prevalence_shift.csv", prev),
                        ("bootstrap_candidates.csv", candidates),
                        ("label_audit_metrics.csv", labels), ("control_metrics.csv", ctrl)]:
        table.to_csv(args.output / name, index=False)

    components = df.groupby("component_id").size()
    summary = {
        "status": "checked",
        "rows": len(df), "categories": sorted(df.category.unique()), "folds": sorted(df.fold.unique()),
        "positive": int(df.target.sum()), "negative": int((1-df.target).sum()),
        "components": int(len(components)), "multi_edge_components": int((components > 1).sum()),
        "max_component_edges": int(components.max()), "fashion_audited_rows": int(df.audited_label.notna().sum()),
        "baseline": BASE, "alternatives": ALTS, "bootstrap_reps_requested": args.bootstrap,
        "seed": args.seed, "raw_fashion_audit_records": len(raw_audit),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    logging.info("complete summary=%s", json.dumps(summary, ensure_ascii=False))
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
