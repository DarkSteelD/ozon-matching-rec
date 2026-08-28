"""Hourly matching LB refresh: commit, push, then notify Telegram on change."""
from __future__ import annotations

import hashlib
import html
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from pull_public_lb import cookie, our_user_id

REPO = Path(__file__).resolve().parents[2]
WORKSPACE = REPO.parents[1]
REPORT = REPO / "members/dzkhomidov/reports/ODS_CATEGORY_DETAIL_LATEST.md"
SUBMISSIONS = REPO / "members/darksteeld/reports/ods_submissions.e-cup-2026-matching.json"
LEADERBOARD = REPO / "members/darksteeld/reports/ods_leaderboard.e-cup-2026-matching.json"
STATE = WORKSPACE / "run/matching_lb_last_notified.sha256"
DEADLINE = datetime(2026, 9, 1, tzinfo=ZoneInfo("Europe/Moscow"))


def git(*args: str, capture: bool = False) -> str:
    result = subprocess.run(
        ["git", *args], cwd=REPO, check=True, text=True,
        capture_output=capture,
    )
    return result.stdout.strip() if capture else ""


def report_digest() -> str:
    return hashlib.sha256(REPORT.read_bytes()).hexdigest()


def build_message() -> str:
    submissions = json.loads(SUBMISSIONS.read_text())["submissions"]
    successful = [s for s in submissions if s.get("status") == "success"]
    best = max(successful, key=lambda s: s["metrics"]["total_prauc"])
    baseline = max(
        (s for s in successful if s["id"] != best["id"]),
        key=lambda s: s["metrics"]["total_prauc"],
    )
    leaderboard = json.loads(LEADERBOARD.read_text())["data"]
    uid = our_user_id(cookie())
    ours = next(
        r for r in leaderboard["leaderboard"]
        if r["team"].get("is_member") or uid in [m.get("id") for m in r["team"].get("members", [])]
    )
    old = baseline["metrics"]["per_category_prauc"]
    new = best["metrics"]["per_category_prauc"]
    gains = sorted(((new[c] - old[c], c) for c in new), reverse=True)[:4]
    gain_text = ", ".join(f"{html.escape(c)} <b>{d:+.6f}</b>" for d, c in gains)
    delta = best["metrics"]["total_prauc"] - baseline["metrics"]["total_prauc"]
    commit = git("rev-parse", "--short", "HEAD", capture=True)
    base = "https://github.com/DarkSteelD/ozon-matching-rec/blob/main/members/dzkhomidov/reports"
    return (
        "📊 <b>Matching — ODS обновился</b>\n\n"
        f"Roma Bazuka: <b>{ours['place']} / {leaderboard['teams_total']}</b>, "
        f"macro PR-AUC <b>{ours['metrics']['total_prauc']:.10f}</b>.\n"
        f"Лучший: <code>{html.escape(best['file_name'])}</code>, "
        f"Δ к <code>{html.escape(baseline['file_name'])}</code>: <b>{delta:+.10f}</b>.\n\n"
        f"Лучшие дельты: {gain_text}.\n\n"
        f'<a href="{base}/ODS_CATEGORY_DETAIL_LATEST.md">Детализация</a> · '
        f'<a href="{base}/ODS_CATEGORY_DETAIL_LATEST.csv">CSV</a> · '
        f'<a href="https://github.com/DarkSteelD/ozon-matching-rec/commit/{commit}">commit {commit}</a>\n\n'
        "Конкурсную квоту монитор не расходует."
    )


def notify_if_changed() -> None:
    digest = report_digest()
    if STATE.is_file() and STATE.read_text().strip() == digest:
        print("matching ODS detail: Telegram unchanged")
        return
    sys.path.insert(0, str(WORKSPACE / "platform"))
    from ozonlb.config import TRACKS
    from ozonlb.notify import send

    result = send(build_message(), track=TRACKS["matching"], preview=False)
    if result is None:
        raise RuntimeError("Telegram message was not sent")
    STATE.parent.mkdir(parents=True, exist_ok=True)
    temporary = STATE.with_suffix(".tmp")
    temporary.write_text(digest + "\n")
    temporary.replace(STATE)
    print(f"matching ODS detail: Telegram message_id={result['message_id']}")


def main() -> None:
    if datetime.now(ZoneInfo("Europe/Moscow")) >= DEADLINE:
        print("matching ODS detail: deadline reached")
        return
    git("fetch", "--quiet", "origin", "main")
    if git("status", "--porcelain", capture=True):
        raise RuntimeError("refusing hourly refresh: worktree is dirty")
    before = git("rev-parse", "HEAD", capture=True)
    if before != git("rev-parse", "origin/main", capture=True):
        raise RuntimeError("refusing hourly refresh: main differs from origin/main")
    subprocess.run([sys.executable, str(REPO / "validation/ops/refresh_ods_detail.py")], cwd=REPO, check=True)
    after = git("rev-parse", "HEAD", capture=True)
    if after != before:
        git("push", "origin", "main")
    if git("status", "--porcelain", capture=True):
        raise RuntimeError("hourly refresh left a dirty worktree")
    notify_if_changed()


if __name__ == "__main__":
    main()
