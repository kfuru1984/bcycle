"""
月次レポート生成スクリプト。

data/{country}/level_momentum.parquet を読み込み、
reports/YYYY-MM.md を生成して reports/README.md のインデックスを更新する。

実行:
    python scripts/generate_report.py
    python scripts/generate_report.py --jp-ok true --us-ok true --kr-ok false
"""
from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

ROOT        = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

COUNTRIES = ["jp", "us", "kr"]


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_country(country: str, run_ok: bool = True) -> dict:
    """Load summary stats from level_momentum.parquet."""
    if not run_ok:
        return {"ok": False, "country": country.upper(), "error": "run_cycle failed"}

    path = ROOT / "data" / country / "level_momentum.parquet"
    if not path.exists():
        return {"ok": False, "country": country.upper(), "error": "parquet not found"}

    try:
        df    = pd.read_parquet(path)
        valid = df.dropna(subset=["level"])
        if valid.empty:
            return {"ok": False, "country": country.upper(), "error": "no valid data"}

        last = valid.iloc[-1]
        prev = valid.iloc[-2] if len(valid) >= 2 else None

        level    = float(last["level"])
        momentum = float(last["momentum"]) if pd.notna(last.get("momentum")) else float("nan")
        stage    = str(last["stage"])      if pd.notna(last.get("stage"))    else "-"
        data_end = str(valid.index[-1])[:7]
        data_beg = str(valid.index[0])[:7]

        trend = "->"
        if prev is not None:
            diff = level - float(prev["level"])
            trend = "up" if diff > 0.5 else ("down" if diff < -0.5 else "flat")

        TREND_SYM = {"up": "↑", "down": "↓", "flat": "→", "->": "→"}

        return {
            "ok":       True,
            "country":  country.upper(),
            "stage":    stage,
            "level":    level,
            "momentum": momentum,
            "trend":    TREND_SYM[trend],
            "data_end": data_end,
            "data_beg": data_beg,
            "n_obs":    len(valid),
        }
    except Exception as exc:
        return {"ok": False, "country": country.upper(), "error": str(exc)}


def stage_history_table(country: str, n: int = 6) -> str:
    """Return markdown table of the last n valid months for a country."""
    path = ROOT / "data" / country / "level_momentum.parquet"
    if not path.exists():
        return "*(データなし)*"

    try:
        df    = pd.read_parquet(path)
        valid = df.dropna(subset=["level"]).tail(n)
        if valid.empty:
            return "*(有効データなし)*"

        rows = ["| 月 | Level | Momentum | Stage |", "|---|---|---|---|"]
        for idx, row in valid.iterrows():
            m_str = f"{row['momentum']:+.2f}" if pd.notna(row.get("momentum")) else "-"
            s_str = str(row["stage"]) if pd.notna(row.get("stage")) else "-"
            rows.append(f"| {str(idx)[:7]} | {row['level']:.1f} | {m_str} | {s_str} |")
        return "\n".join(rows)
    except Exception as exc:
        return f"*(エラー: {exc})*"


def fetch_cn_signal() -> str:
    """Fetch the CN BCI signal from FRED. Returns a markdown-formatted string."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
        from bcycle_jp.adapters.fred import FredAdapter

        fred = FredAdapter()
        if not fred.is_available():
            return "**CN Signal:** FRED_API_KEY 未設定"

        s   = fred.fetch({"series_id": "CHNBSCICP02STSAM"}, start=date(2000, 1, 1))
        s   = s.dropna()
        val = float(s.iloc[-1])
        dts = str(s.index[-1])[:7]
        lbl = "Expanding" if val >= 0.0 else "Contracting"
        return (
            f"**CN Signal:** BCI {val:.2f} ({lbl}) [{dts}]  \n"
            f"*(OECD Mfg Business Confidence — NBS/Caixin PMI is not available on FRED/OECD)*"
        )
    except Exception as exc:
        return f"**CN Signal:** 取得失敗 ({exc})"


# ─────────────────────────────────────────────────────────────────────────────
# Report assembly
# ─────────────────────────────────────────────────────────────────────────────

def generate_report(
    run_ok: dict[str, bool] | None = None,
    run_date: date | None = None,
) -> Path:
    """Generate reports/YYYY-MM.md and return the path."""
    if run_date is None:
        run_date = date.today()
    if run_ok is None:
        run_ok = {c: True for c in COUNTRIES}

    report_month = run_date.strftime("%Y-%m")
    out_path     = REPORTS_DIR / f"{report_month}.md"

    country_data = {c: load_country(c, run_ok.get(c, True)) for c in COUNTRIES}
    cn_line      = fetch_cn_signal()

    # --- Summary table ---
    summary_rows = []
    for c in COUNTRIES:
        d = country_data[c]
        if d["ok"]:
            summary_rows.append(
                f"| {d['country']} | {d['stage']} | {d['level']:.1f} "
                f"| {d['momentum']:+.2f} | {d['trend']} |"
            )
        else:
            summary_rows.append(f"| {d['country']} | 取得失敗 | — | — | — |")

    summary_block = (
        "| 国 | ステージ | Level | Momentum | 前月比 |\n"
        "|---|---|---|---|---|\n"
        + "\n".join(summary_rows)
    )

    # --- Per-country stage history ---
    history_parts = []
    for c in COUNTRIES:
        history_parts.append(f"### {c.upper()}\n\n{stage_history_table(c)}")
    history_block = "\n\n".join(history_parts)

    # --- Data coverage ---
    cov_lines = []
    for c in COUNTRIES:
        d = country_data[c]
        if d["ok"]:
            cov_lines.append(
                f"- **{d['country']}**: {d['data_beg']} ~ {d['data_end']} ({d['n_obs']} obs)"
            )
        else:
            cov_lines.append(f"- **{d['country']}**: {d.get('error', '不明')}")
    cov_block = "\n".join(cov_lines)

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    report = f"""\
# Business Cycle Report: {report_month}

*生成: {generated_at}*

## サマリー

{summary_block}

{cn_line}

## ステージ推移（直近6ヶ月）

{history_block}

## チャート

![JP](../data/jp/stage_timeline.png)
![US](../data/us/stage_timeline.png)
![KR](../data/kr/stage_timeline.png)

## データカバレッジ

{cov_block}
"""

    out_path.write_text(report, encoding="utf-8")
    print(f"レポート生成: {out_path}")
    return out_path


def update_readme(report_path: Path) -> None:
    """Prepend a link to the new report in reports/README.md."""
    readme  = REPORTS_DIR / "README.md"
    stem    = report_path.stem          # e.g. "2026-05"
    new_line = f"- [{stem}](./{report_path.name})\n"

    if readme.exists():
        content = readme.read_text(encoding="utf-8")
        if new_line in content:
            return  # already listed
        # Insert after the first blank line following the header
        lines = content.splitlines(keepends=True)
        insert_at = len(lines)
        for i, line in enumerate(lines):
            if line.startswith("- ["):
                insert_at = i
                break
        lines.insert(insert_at, new_line)
        readme.write_text("".join(lines), encoding="utf-8")
    else:
        readme.write_text(
            "# Business Cycle Reports\n\n"
            "## Index\n\n"
            f"{new_line}",
            encoding="utf-8",
        )

    print(f"README 更新: {readme}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="月次レポート生成")
    parser.add_argument("--jp-ok", default="true",
                        help="JP run succeeded? (true/false, default true)")
    parser.add_argument("--us-ok", default="true",
                        help="US run succeeded? (true/false, default true)")
    parser.add_argument("--kr-ok", default="true",
                        help="KR run succeeded? (true/false, default true)")
    args = parser.parse_args()

    def _bool(s: str) -> bool:
        return s.lower() in ("true", "1", "yes", "success")

    run_ok = {
        "jp": _bool(args.jp_ok),
        "us": _bool(args.us_ok),
        "kr": _bool(args.kr_ok),
    }

    rpt = generate_report(run_ok=run_ok)
    update_readme(rpt)
