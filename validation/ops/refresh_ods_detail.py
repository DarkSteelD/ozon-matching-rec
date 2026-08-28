"""Refresh matching ODS detail and commit it when the remote snapshot changed."""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from pull_public_lb import cookie, our_user_id, render as render_public

REPO = Path(__file__).resolve().parents[2]
WORKSPACE = REPO.parents[1]
FETCHER = WORKSPACE / "repos/ozon-ltv/members/darksteeld/scripts/ods_submissions.py"
TRACK = "e-cup-2026-matching"
RAW_REPORTS = REPO / "members/darksteeld/reports"
DETAIL_REPORTS = REPO / "members/dzkhomidov/reports"
MANAGED = [
    REPO / "PUBLIC_LEADERBOARD.md",
    RAW_REPORTS / f"SUBMISSIONS.{TRACK}.md",
    RAW_REPORTS / f"ods_leaderboard.{TRACK}.json",
    RAW_REPORTS / f"ods_submissions.{TRACK}.json",
    DETAIL_REPORTS / "ODS_CATEGORY_DETAIL_LATEST.csv",
    DETAIL_REPORTS / "ODS_CATEGORY_DETAIL_LATEST.md",
]


def score(submission: dict) -> float:
    return float(submission["metrics"]["total_prauc"])


def render_detail(submissions: dict, leaderboard: dict) -> tuple[str, list[list[object]]]:
    successful = [
        s for s in submissions["submissions"]
        if s.get("status") == "success" and isinstance((s.get("metrics") or {}).get("total_prauc"), (int, float))
    ]
    latest = max(successful, key=lambda s: s["created_at"])
    best = max(successful, key=score)
    baseline = max((s for s in successful if s["id"] != best["id"]), key=score)

    uid = our_user_id(cookie())
    rows = leaderboard["data"]["leaderboard"]
    ours = next(r for r in rows if uid in [m.get("id") for m in r["team"].get("members", [])])
    leader = rows[0]
    latest_cats = best["metrics"]["per_category_prauc"]
    baseline_cats = baseline["metrics"]["per_category_prauc"]
    categories = sorted(latest_cats)
    delta = score(best) - score(baseline)
    now = datetime.now(timezone.utc).astimezone()

    table = [[category, baseline_cats[category], latest_cats[category], latest_cats[category] - baseline_cats[category]] for category in categories]
    lines = [
        "# Matching: актуальная детализация ODS по категориям",
        "",
        f"Срез: {now:%Y-%m-%d %H:%M %Z}. Команда `{ours['team']['name']}`: "
        f"**{ours['place']} / {leaderboard['data']['teams_total']}**, score **{ours['metrics']['total_prauc']:.10f}**. "
        f"До лидера: **{leader['metrics']['total_prauc'] - ours['metrics']['total_prauc']:.10f}**.",
        "",
        f"Текущий лучший: `{best['file_name']}`, ODS ID `{best['id']}`, score **{score(best):.10f}**. "
        f"Прирост к следующему лучшему `{baseline['file_name']}`: **{delta:+.10f}**. Вердикт: **GO**.",
        f"Последняя успешная посылка: `{latest['file_name']}`, score **{score(latest):.10f}**, "
        f"Δ к текущему лучшему **{score(latest) - score(best):+.10f}** "
        f"({'GO' if latest['id'] == best['id'] else 'NO_GO'}).",
        "",
        f"| Категория | `{baseline['file_name']}` | `{best['file_name']}` | Δ |",
        "|---|---:|---:|---:|",
    ]
    lines.extend(f"| {c} | {old:.6f} | {new:.6f} | {d:+.6f} |" for c, old, new, d in table)
    lines.append("")
    return "\n".join(lines), table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-commit", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="matching-lb-") as temp:
        out = Path(temp)
        subprocess.run([sys.executable, str(FETCHER), "--track", TRACK, "--out", str(out)], check=True)
        new_submissions = out / f"ods_submissions.{TRACK}.json"
        new_leaderboard = out / f"ods_leaderboard.{TRACK}.json"
        current_submissions = RAW_REPORTS / new_submissions.name
        current_leaderboard = RAW_REPORTS / new_leaderboard.name
        changed = args.force or not (
            current_submissions.is_file()
            and current_leaderboard.is_file()
            and new_submissions.read_bytes() == current_submissions.read_bytes()
            and new_leaderboard.read_bytes() == current_leaderboard.read_bytes()
            and all(path.is_file() for path in MANAGED)
        )
        if not changed:
            print("matching ODS detail: unchanged")
            return

        submissions = json.loads(new_submissions.read_text())
        leaderboard = json.loads(new_leaderboard.read_text())
        detail_md, category_rows = render_detail(submissions, leaderboard)
        RAW_REPORTS.mkdir(parents=True, exist_ok=True)
        DETAIL_REPORTS.mkdir(parents=True, exist_ok=True)
        for name in (f"SUBMISSIONS.{TRACK}.md", new_leaderboard.name, new_submissions.name):
            shutil.copyfile(out / name, RAW_REPORTS / name)
        (REPO / "PUBLIC_LEADERBOARD.md").write_text(
            render_public(leaderboard["data"], our_user_id(cookie())), encoding="utf-8"
        )
        (DETAIL_REPORTS / "ODS_CATEGORY_DETAIL_LATEST.md").write_text(detail_md, encoding="utf-8")
        with (DETAIL_REPORTS / "ODS_CATEGORY_DETAIL_LATEST.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["category", "baseline", "latest", "delta"])
            writer.writerows(category_rows)

    if args.no_commit:
        print("matching ODS detail: refreshed (commit skipped)")
        return
    subprocess.run(["git", "add", "--", *[str(path.relative_to(REPO)) for path in MANAGED]], cwd=REPO, check=True)
    staged = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=REPO)
    if staged.returncode == 1:
        subprocess.run(["git", "commit", "-m", "Refresh matching ODS leaderboard detail"], cwd=REPO, check=True)
        print("matching ODS detail: refreshed and committed")
    elif staged.returncode == 0:
        print("matching ODS detail: no staged changes")
    else:
        raise SystemExit(staged.returncode)


if __name__ == "__main__":
    main()
