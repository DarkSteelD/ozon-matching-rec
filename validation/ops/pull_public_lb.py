"""Pull the REAL competition leaderboard from ODS and render PUBLIC_LEADERBOARD.md.

This is deliberately separate from the internal validation (CV) leaderboard:
  * validation/leaderboard.md  = наши эксперименты на замороженных CV-фолдах
    (наша собственная валидация; публичный скор наших посылок — лишь диагностика).
  * PUBLIC_LEADERBOARD.md      = реальные места ВСЕХ команд на ODS (это файл).

Needs the ODS cookie (~/.config/ecup-agent/ods_cookie); read-only. Config from
the co-located ops_config.json (track_slug). Run standalone or via run_all.

    python validation/ops/pull_public_lb.py [--commit]
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import requests

OPS_DIR = Path(__file__).resolve().parent
REPO = OPS_DIR.parents[1]
CFG = json.loads((OPS_DIR / "ops_config.json").read_text())
TRACK = CFG["track_slug"]
COOKIE_FILE = Path.home() / ".config" / "ecup-agent" / "ods_cookie"
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/150.0.0.0 Safari/537.36"


def cookie() -> str:
    if os.environ.get("ODS_COOKIE", "").strip():
        return os.environ["ODS_COOKIE"].strip()
    return COOKIE_FILE.read_text().strip()


def our_user_id(ck: str) -> str | None:
    m = re.search(r"access_token=([^;\s]+)", ck)
    if not m:
        return None
    payload = m.group(1).split(".")[1]
    payload += "=" * (-len(payload) % 4)
    return json.loads(base64.urlsafe_b64decode(payload)).get("user_id")


def fetch_lb(ck: str) -> dict:
    page = requests.get(f"https://ods.ai/competitions/{TRACK}", timeout=30,
                        headers={"User-Agent": UA, "Cookie": ck})
    build = re.search(r'"buildId":"([^"]+)"', page.text).group(1)
    url = (f"https://ods.ai/_next/data/{build}/competitions/{TRACK}/"
           f"leaderboard.json?competitionId={TRACK}")
    r = requests.get(url, timeout=30, headers={"User-Agent": UA, "Cookie": ck})
    r.raise_for_status()
    return r.json()["pageProps"]["leaderboard"]["data"]


def render(d: dict, uid: str | None) -> str:
    metric = d["metrics"][0]
    slug, title, rev = metric["slug"], metric["title"], metric["is_reversed"]
    rows = d["leaderboard"]
    total = d.get("teams_total", len(rows))

    def score(r):
        return r["metrics"].get(slug)

    ours = None
    for r in rows:
        if uid and uid in [m.get("id") for m in r["team"].get("members", [])]:
            ours = r
            break
        if r["team"].get("is_member"):
            ours = r
            break

    out = [f"# Реальный ЛБ соревнования (ODS) — {TRACK}", "",
           f"Публичный лидерборд **всех команд**, снят {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}. "
           f"Метрика **{title}** (`{slug}`, {'меньше = лучше' if rev else 'больше = лучше'}). "
           f"Команд всего: **{total}**.", "",
           "> Это НЕ наша валидация. Наша внутренняя CV-таблица — отдельно, в "
           "`validation/leaderboard.md` и `validation/TOP5.md`.", ""]

    if ours:
        s = score(ours)
        top1 = score(rows[0])
        gap1 = abs(s - top1)
        line = (f"**Наша команда «{ours['team']['name']}»: место {ours['place']} / {total}**, "
                f"{title} {s:.6f}. Отрыв от #1: {gap1:+.6f}")
        if len(rows) > 1:
            line += f"; от #2: {abs(s - score(rows[1])):+.6f}"
        out += [line, ""]
    else:
        out += [f"_Нашу команду не видно в топ-{len(rows)} (всего {total}); "
                "проверь свежую куку или что команда засабмитила._", ""]

    out += [f"| место | команда | {title} | посылок |", "|---:|---|---:|---:|"]
    shown = rows[:15]
    if ours and ours not in shown:
        shown = shown + [ours]
    for r in shown:
        mark = " ⬅ мы" if r is ours else ""
        sc = score(r)
        out.append(f"| {r['place']} | {r['team']['name']}{mark} | "
                   f"{sc:.6f} | {r['team'].get('submissions_count','')} |")
    out.append("")
    return "\n".join(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--commit", action="store_true")
    args = ap.parse_args()

    ck = cookie()
    data = fetch_lb(ck)
    md = render(data, our_user_id(ck))
    dest = REPO / "PUBLIC_LEADERBOARD.md"
    dest.write_text(md, encoding="utf-8")
    print(f"[public-lb] wrote {dest.relative_to(REPO)}")

    if args.commit:
        subprocess.run(["git", "add", "PUBLIC_LEADERBOARD.md"], cwd=REPO)
        st = subprocess.run(["git", "status", "--porcelain", "PUBLIC_LEADERBOARD.md"],
                            cwd=REPO, capture_output=True, text=True)
        if st.stdout.strip():
            subprocess.run(["git", "commit", "-q", "-m",
                            f"Refresh real ODS leaderboard ({TRACK})\n\n"
                            "Automated by validation/ops/pull_public_lb.py.\n\n"
                            "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"], cwd=REPO)
            subprocess.run(["git", "pull", "--rebase", "--quiet"], cwd=REPO)
            subprocess.run(["git", "push", "--quiet"], cwd=REPO)
            print("[public-lb] committed + pushed")
        else:
            print("[public-lb] unchanged")
