#!/usr/bin/env python3
"""Label-free pair-graph feature audit with an out-of-fold meta-model."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score


ROOT = Path("/home/dzkhomidov/matching-work/rescue_20260824/graph_matrices")
INPUT = Path("/home/dzkhomidov/matching-work/rescue_20260824/hard_negative/hand_features.parquet")
SEED = 20260824


class DSU:
    def __init__(self, n: int):
        self.p = np.arange(n, dtype=np.int32)
        self.sz = np.ones(n, dtype=np.int32)

    def find(self, x: int) -> int:
        p = self.p
        while p[x] != x:
            p[x] = p[p[x]]
            x = int(p[x])
        return x

    def union(self, a: int, b: int) -> None:
        a, b = self.find(a), self.find(b)
        if a == b:
            return
        if self.sz[a] < self.sz[b]:
            a, b = b, a
        self.p[b] = a
        self.sz[a] += self.sz[b]


def top_two_update(best: np.ndarray, second: np.ndarray, idx: int, value: float) -> None:
    if value >= best[idx]:
        second[idx] = best[idx]
        best[idx] = value
    elif value > second[idx]:
        second[idx] = value


def other_max(best: np.ndarray, second: np.ndarray, idx: int, value: float, degree: np.ndarray) -> float:
    if degree[idx] <= 1:
        return np.nan
    return float(second[idx] if value >= best[idx] else best[idx])


def build_features(df: pl.DataFrame) -> pd.DataFrame:
    a = df["id1"].to_numpy()
    b = df["id2"].to_numpy()
    score = df["ce_oof"].to_numpy().astype(np.float64)
    ids = np.unique(np.concatenate([a, b]))
    u = np.searchsorted(ids, a).astype(np.int32)
    v = np.searchsorted(ids, b).astype(np.int32)
    n = len(ids)
    dsu = DSU(n)
    degree = np.zeros(n, dtype=np.int32)
    score_sum = np.zeros(n, dtype=np.float64)
    best = np.full(n, -np.inf, dtype=np.float64)
    second = np.full(n, -np.inf, dtype=np.float64)
    adjacency: dict[int, set[int]] = defaultdict(set)
    for x, y, s in zip(u, v, score):
        dsu.union(int(x), int(y))
        degree[x] += 1
        degree[y] += 1
        score_sum[x] += s
        score_sum[y] += s
        top_two_update(best, second, int(x), float(s))
        top_two_update(best, second, int(y), float(s))
        adjacency[int(x)].add(int(y))
        adjacency[int(y)].add(int(x))

    roots = np.fromiter((dsu.find(i) for i in range(n)), dtype=np.int32, count=n)
    comp_nodes = np.bincount(roots, minlength=n).astype(np.int32)
    edge_root = roots[u]
    comp_edges = np.bincount(edge_root, minlength=n).astype(np.int32)
    comp_score_sum = np.bincount(edge_root, weights=score, minlength=n)

    out = defaultdict(list)
    for x, y, s, r in zip(u, v, score, edge_root):
        nx, ny = adjacency[int(x)], adjacency[int(y)]
        small, large = (nx, ny) if len(nx) <= len(ny) else (ny, nx)
        common_nodes = [z for z in small if z in large]
        common = len(common_nodes)
        union = degree[x] + degree[y] - common
        adamic = sum(1.0 / math.log(max(2, int(degree[z]))) for z in common_nodes)
        du, dv = int(degree[x]), int(degree[y])
        omu = (score_sum[x] - s) / (du - 1) if du > 1 else np.nan
        omv = (score_sum[y] - s) / (dv - 1) if dv > 1 else np.nan
        ou = other_max(best, second, int(x), float(s), degree)
        ov = other_max(best, second, int(y), float(s), degree)
        ce = int(comp_edges[r])
        cn = int(comp_nodes[r])
        out["deg_min"].append(min(du, dv))
        out["deg_max"].append(max(du, dv))
        out["log_pref_attach"].append(math.log1p(du * dv))
        out["comp_nodes"].append(cn)
        out["comp_edges"].append(ce)
        out["cyclomatic"].append(ce - cn + 1)
        out["common_neighbors"].append(common)
        out["neighbor_jaccard"].append(common / union if union else 0.0)
        out["adamic_adar"].append(adamic)
        out["other_mean_min"].append(np.nanmin([omu, omv]) if not (np.isnan(omu) and np.isnan(omv)) else np.nan)
        out["other_mean_max"].append(np.nanmax([omu, omv]) if not (np.isnan(omu) and np.isnan(omv)) else np.nan)
        out["other_max_min"].append(np.nanmin([ou, ov]) if not (np.isnan(ou) and np.isnan(ov)) else np.nan)
        out["other_max_max"].append(np.nanmax([ou, ov]) if not (np.isnan(ou) and np.isnan(ov)) else np.nan)
        out["component_other_mean"].append((comp_score_sum[r] - s) / (ce - 1) if ce > 1 else np.nan)
        out["score_minus_neighbor"].append(s - np.nanmean([omu, omv]) if not (np.isnan(omu) and np.isnan(omv)) else 0.0)
    return pd.DataFrame(out)


def macro_ap(y: np.ndarray, pred: np.ndarray, cat: np.ndarray) -> float:
    return float(np.mean([average_precision_score(y[cat == c], pred[cat == c]) for c in np.unique(cat)]))


def shuffled_graph(x: pd.DataFrame, fold: np.ndarray, cat: np.ndarray) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    z = x.copy()
    degree_bin = np.minimum(5, np.floor(np.log2(np.maximum(1, x["deg_max"].to_numpy())))).astype(np.int8)
    keys = pd.DataFrame({"fold": fold, "cat": cat, "degree_bin": degree_bin})
    for _, idx in keys.groupby(["fold", "cat", "degree_bin"], sort=False).groups.items():
        idx = np.asarray(list(idx), dtype=np.int64)
        if len(idx) > 1:
            z.iloc[idx] = x.iloc[rng.permutation(idx)].to_numpy()
    return z


def fit_oof(df: pl.DataFrame, graph: pd.DataFrame) -> dict[str, np.ndarray]:
    y = df["target"].to_numpy().astype(np.int8)
    score = df["ce_oof"].to_numpy().astype(np.float64)
    fold = df["fold"].to_numpy()
    cat = df["category"].to_numpy()
    shuffled = shuffled_graph(graph, fold, cat)
    preds = {"baseline": score.copy(), "graph": np.zeros(len(y)), "shuffled": np.zeros(len(y))}
    for held in sorted(np.unique(fold)):
        va_fold = fold == held
        tr_fold = ~va_fold
        for c in np.unique(cat):
            tr = tr_fold & (cat == c)
            va = va_fold & (cat == c)
            if not va.any():
                continue
            for name, gx in [("graph", graph), ("shuffled", shuffled)]:
                xtr = np.column_stack([score[tr], gx.loc[tr].to_numpy()])
                xva = np.column_stack([score[va], gx.loc[va].to_numpy()])
                model = HistGradientBoostingClassifier(
                    learning_rate=0.05,
                    max_iter=80,
                    max_leaf_nodes=7,
                    min_samples_leaf=200,
                    l2_regularization=10.0,
                    random_state=SEED,
                )
                model.fit(xtr, y[tr])
                preds[name][va] = model.predict_proba(xva)[:, 1]
    return preds


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    df = pl.read_parquet(INPUT).select("fold", "id1", "id2", "target", "category", "ce_oof")
    graph = build_features(df)
    feature_path = ROOT / "graph_features.parquet"
    pl.from_pandas(graph).write_parquet(feature_path)
    preds = fit_oof(df, graph)
    y = df["target"].to_numpy().astype(np.int8)
    fold = df["fold"].to_numpy()
    cat = df["category"].to_numpy()
    metrics = {}
    for name, pred in preds.items():
        metrics[name] = {
            "macro_all": macro_ap(y, pred, cat),
            "pooled_all": float(average_precision_score(y, pred)),
            "folds": {
                f: {
                    "macro": macro_ap(y[fold == f], pred[fold == f], cat[fold == f]),
                    "pooled": float(average_precision_score(y[fold == f], pred[fold == f])),
                }
                for f in sorted(np.unique(fold))
            },
        }
    metrics["coverage"] = {
        "rows": len(y),
        "multi_edge_component_rows": int((graph["comp_edges"].to_numpy() > 1).sum()),
        "multi_edge_component_fraction": float((graph["comp_edges"].to_numpy() > 1).mean()),
        "triangle_rows": int((graph["common_neighbors"].to_numpy() > 0).sum()),
    }
    metrics["deltas"] = {
        "graph_minus_baseline_macro": metrics["graph"]["macro_all"] - metrics["baseline"]["macro_all"],
        "shuffled_minus_baseline_macro": metrics["shuffled"]["macro_all"] - metrics["baseline"]["macro_all"],
        "graph_minus_shuffled_macro": metrics["graph"]["macro_all"] - metrics["shuffled"]["macro_all"],
    }
    (ROOT / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2))
    out = df.select("fold", "id1", "id2", "target", "category").with_columns(
        *[pl.Series(name, pred) for name, pred in preds.items()]
    )
    out.write_parquet(ROOT / "oof_predictions.parquet")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
