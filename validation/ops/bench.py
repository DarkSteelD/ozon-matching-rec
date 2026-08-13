"""Nightly internal bench for one E-CUP track repo (portable, host-agnostic).

Same file in every track repo at `validation/ops/bench.py`; behaviour is driven
by the co-located `ops_config.json`, so a repo self-serves: clone, then

    python validation/ops/bench.py            # rebuild + TOP5 + MLflow, no push
    python validation/ops/bench.py --commit   # + git add/commit/pull --rebase/push

Steps: (1) rebuild the leaderboard from committed result JSONs via the repo's own
`make leaderboard`; (2) regenerate `validation/TOP5.md` (top-5 by the primary
metric); (3) log every result JSON to a local MLflow store (one experiment per
track); (4) optionally commit and push leaderboards + TOP5 + results.

Designed to run at 23:00 on whatever host holds the predictions (a member's box
now, a shared vast instance later). Deploy line for later (host TBD, not
installed here):

    0 23 * * * cd <repo> && <py> validation/ops/bench.py --commit >> bench.log 2>&1
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

OPS_DIR = Path(__file__).resolve().parent
REPO = OPS_DIR.parents[1]                      # validation/ops/ -> repo root
CFG = json.loads((OPS_DIR / "ops_config.json").read_text())
PRIMARY = CFG["primary"]
REVERSED = CFG["reversed"]                     # True = lower is better
TRACK = CFG["track_slug"]
PUBLIC_COL = CFG.get("public_col", "")
METRIC_FIELDS = CFG.get("metric_fields", [PRIMARY])


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=REPO, text=True, capture_output=True, **kw)


def rebuild_leaderboard(py: str) -> None:
    r = run(["make", "leaderboard", f"PY={py}"])
    if r.returncode != 0:                       # fall back to the raw modules
        run([py, "-m", "validation.evaluate", "--rebuild-only"])
        run([py, "-m", "validation.render_leaderboard"])


def read_leaderboard() -> list[dict]:
    f = REPO / CFG["leaderboard"]
    if not f.exists():
        return []
    with f.open() as fh:
        return list(csv.DictReader(fh))


def write_top5(rows: list[dict]) -> Path:
    def key(r):
        try:
            v = float(r[PRIMARY])
        except (KeyError, ValueError):
            v = float("inf") if REVERSED else float("-inf")
        return v
    ranked = sorted(rows, key=key, reverse=not REVERSED)[:5]
    arrow = "меньше = лучше" if REVERSED else "больше = лучше"
    out = [f"# TOP-5 — {TRACK}", "",
           f"Обновлено бенчем {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}. "
           f"Первичная метрика `{PRIMARY}` ({arrow}). "
           f"Источник: `{CFG['leaderboard']}`. Не редактировать руками.", "",
           f"| # | member | experiment | {PRIMARY} | public | notes |",
           "|---:|---|---|---:|---:|---|"]
    for i, r in enumerate(ranked, 1):
        pub = r.get(PUBLIC_COL, "") or "—"
        note = (r.get("notes", "") or "").replace("|", "/")[:70]
        out.append(f"| {i} | {r.get('member','')} | {r.get('experiment','')} | "
                   f"{r.get(PRIMARY,'')} | {pub} | {note} |")
    out.append("")
    dest = REPO / CFG["top5_out"]
    dest.write_text("\n".join(out), encoding="utf-8")
    return dest


def log_mlflow() -> int:
    try:
        import mlflow
    except ImportError:
        print("[mlflow] not installed — skipping (pip install mlflow)")
        return 0
    # MLflow 3.x deprecated the file store; use a sqlite backend (portable now,
    # swappable for a shared server URI later via ECUP_MLFLOW_URI).
    uri = os.environ.get("ECUP_MLFLOW_URI")
    if not uri:
        db = os.path.expanduser("~/.config/ecup-agent/mlflow.db")
        Path(db).parent.mkdir(parents=True, exist_ok=True)
        uri = f"sqlite:///{db}"
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(TRACK)
    n = 0
    for jf in sorted((REPO / "validation" / "results").glob("*/*.json")):
        d = json.loads(jf.read_text())
        member, exp = d.get("member", jf.parent.name), d.get("experiment", jf.stem)
        run_name = f"{member}/{exp}"
        # idempotent: one run per member/experiment, overwrite by deleting dupes
        existing = mlflow.search_runs(
            filter_string=f"tags.mlflow.runName = '{run_name}'",
            output_format="list")
        for old in existing:
            mlflow.delete_run(old.info.run_id)
        with mlflow.start_run(run_name=run_name):
            mlflow.set_tags({"member": member, "experiment": exp,
                             "commit": d.get("commit", ""), "track": TRACK,
                             "fold_spec_version": str(d.get("fold_spec_version", "")),
                             "notes": (d.get("notes", "") or "")[:480]})
            for fld in METRIC_FIELDS:
                if isinstance(d.get(fld), (int, float)):
                    mlflow.log_metric(fld, float(d[fld]))
            for fi, fold in enumerate(d.get("folds", []) or [], 1):
                if isinstance(fold, dict):
                    for k, v in fold.items():
                        if isinstance(v, (int, float)):
                            mlflow.log_metric(f"fold{fi}_{k}", float(v))
            pub = d.get(PUBLIC_COL) or d.get("public_" + PRIMARY.replace("mean_", ""))
            if isinstance(pub, (int, float)):
                mlflow.log_metric("public", float(pub))
            n += 1
    print(f"[mlflow] logged {n} runs to {uri} (experiment '{TRACK}')")
    return n


def git_commit_push() -> None:
    paths = [CFG["leaderboard"], CFG["leaderboard"].replace(".csv", ".md"),
             CFG["top5_out"], "validation/results"]
    run(["git", "add", *paths])
    st = run(["git", "status", "--porcelain", *paths])
    if not st.stdout.strip():
        print("[git] nothing changed")
        return
    msg = (f"Nightly bench: refresh leaderboard + TOP5 ({TRACK})\n\n"
           "Automated by validation/ops/bench.py.\n\n"
           "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>")
    c = run(["git", "commit", "-m", msg])
    if c.returncode != 0:
        print(f"[git] commit failed: {c.stderr[:200]}")
        return
    run(["git", "pull", "--rebase", "--quiet"])
    p = run(["git", "push", "--quiet"])
    print("[git] pushed" if p.returncode == 0 else f"[git] push failed: {p.stderr[:200]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--commit", action="store_true", help="git add/commit/push results")
    ap.add_argument("--no-mlflow", action="store_true")
    ap.add_argument("--py", default=sys.executable, help="python for make leaderboard")
    args = ap.parse_args()

    rebuild_leaderboard(args.py)
    rows = read_leaderboard()
    dest = write_top5(rows)
    print(f"[top5] {dest.relative_to(REPO)} ({min(len(rows),5)} of {len(rows)} rows)")
    if not args.no_mlflow:
        log_mlflow()
    if args.commit:
        git_commit_push()
    else:
        print("[dry] pass --commit to push leaderboards + TOP5")
