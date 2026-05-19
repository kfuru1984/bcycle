"""
横断ダッシュボード生成スクリプト。

読み込み:
  data/{jp,us,kr}/level_momentum.parquet
  data/{jp,us,kr}/stage_timeline.png
  data/jp/stage_factor_heatmap.png
  data/us/stage_sector_heatmap.png

出力: docs/index.html (自己完結 HTML、画像は base64 埋め込み)

実行:
    python scripts/generate_dashboard.py
"""
from __future__ import annotations

import base64
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

ROOT     = Path(__file__).resolve().parents[1]
DOCS_DIR = ROOT / "docs"
DOCS_DIR.mkdir(exist_ok=True)

COUNTRIES = ["jp", "us", "kr"]

STAGE_COLORS: dict[str, str] = {
    "回復": "#4a9eff",
    "上昇": "#00c853",
    "成熟": "#ffab00",
    "軟化": "#ff6d00",
    "下降": "#f44336",
}

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _b64(path: Path) -> str | None:
    """Return base64 data URI for a PNG, or None if file is missing."""
    if not path.exists():
        return None
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def _load_country(country: str) -> dict:
    """Load the most recent cycle stats from level_momentum.parquet."""
    path = ROOT / "data" / country / "level_momentum.parquet"
    if not path.exists():
        return {"ok": False, "country": country.upper()}
    try:
        df    = pd.read_parquet(path)
        valid = df.dropna(subset=["level"])
        if valid.empty:
            return {"ok": False, "country": country.upper()}

        last = valid.iloc[-1]
        prev = valid.iloc[-2] if len(valid) >= 2 else None

        level    = float(last["level"])
        momentum = float(last["momentum"]) if pd.notna(last.get("momentum")) else 0.0
        stage    = str(last["stage"])      if pd.notna(last.get("stage"))    else "-"
        date_str = str(valid.index[-1])[:7]

        trend = "→"
        if prev is not None:
            diff = level - float(prev["level"])
            trend = "↑" if diff > 0.5 else ("↓" if diff < -0.5 else "→")

        return {
            "ok":       True,
            "country":  country.upper(),
            "stage":    stage,
            "level":    level,
            "momentum": momentum,
            "trend":    trend,
            "date":     date_str,
            "color":    STAGE_COLORS.get(stage, "#888888"),
        }
    except Exception as exc:
        return {"ok": False, "country": country.upper(), "error": str(exc)}


def _cn_signal_html() -> str:
    """Return CN signal as an HTML snippet. Reads from local cache if available."""
    try:
        cache = ROOT / "data" / "fred_CHNBSCICP02STSAM.parquet"
        if cache.exists():
            s   = pd.read_parquet(cache).squeeze().dropna().sort_index()
            val = float(s.iloc[-1])
            dts = str(s.index[-1])[:7]
        else:
            from dotenv import load_dotenv
            load_dotenv()
            from bcycle_jp.adapters.fred import FredAdapter
            fred = FredAdapter()
            if not fred.is_available():
                return '<span style="color:#666">CN Signal: FRED_API_KEY 未設定</span>'
            s   = fred.fetch({"series_id": "CHNBSCICP02STSAM"}, start=date(2000, 1, 1)).dropna()
            val = float(s.iloc[-1])
            dts = str(s.index[-1])[:7]

        color = "#00c853" if val >= 0.0 else "#f44336"
        label = "Expanding" if val >= 0.0 else "Contracting"
        return (
            f'<span class="cn-label">CN Signal</span>'
            f'<span class="cn-value" style="color:{color}">'
            f'BCI {val:+.2f} ({label})</span>'
            f'<span class="cn-date">[{dts}]</span>'
            f'<span class="cn-note">OECD Mfg Business Confidence</span>'
        )
    except Exception as exc:
        return f'<span class="cn-label">CN Signal</span><span style="color:#666">取得失敗: {exc}</span>'


# ─────────────────────────────────────────────────────────────────────────────
# HTML fragments
# ─────────────────────────────────────────────────────────────────────────────

def _country_card(d: dict) -> str:
    if not d["ok"]:
        return f"""\
<div class="card card-error">
  <div class="card-country">{d['country']}</div>
  <div class="card-error-msg">データ取得失敗</div>
</div>"""

    color = d["color"]
    mom   = d["momentum"]
    mom_color = "#00c853" if mom > 0.05 else ("#f44336" if mom < -0.05 else "#aaaaaa")
    mom_str   = f"{mom:+.2f}"

    # level bar: 0-100 → width %
    level_pct = min(100, max(0, d["level"]))

    return f"""\
<div class="card" style="border-top:3px solid {color}">
  <div class="card-country">{d['country']}</div>
  <div class="stage-badge" style="background:{color}22;color:{color};border:1px solid {color}55">
    {d['stage']}
  </div>
  <div class="level-track">
    <div class="level-fill" style="width:{level_pct:.1f}%;background:{color}"></div>
  </div>
  <div class="card-stats">
    <div class="stat-item">
      <span class="stat-label">Level</span>
      <span class="stat-value">{d['level']:.1f}</span>
    </div>
    <div class="stat-item">
      <span class="stat-label">Momentum</span>
      <span class="stat-value" style="color:{mom_color}">{mom_str}</span>
    </div>
    <div class="stat-item">
      <span class="stat-label">前月比</span>
      <span class="stat-value">{d['trend']}</span>
    </div>
  </div>
  <div class="card-date">{d['date']}</div>
</div>"""


def _img_card(label: str, uri: str | None, extra_class: str = "") -> str:
    if uri is None:
        return f"""\
<div class="img-card {extra_class}">
  <div class="img-label">{label}</div>
  <div class="img-missing">画像なし</div>
</div>"""
    return f"""\
<div class="img-card {extra_class}">
  <div class="img-label">{label}</div>
  <img src="{uri}" alt="{label}">
</div>"""


# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────

_CSS = """\
*{box-sizing:border-box;margin:0;padding:0}
body{background:#1a1a2e;color:#e0e0e0;font-family:'Segoe UI',system-ui,sans-serif;
     line-height:1.5;padding:20px 24px}
a{color:#4a9eff}
.container{max-width:1400px;margin:0 auto}

/* ── Header ─────────────────────────────────────────────────── */
header{display:flex;align-items:center;justify-content:space-between;
       padding:20px 0 28px;flex-wrap:wrap;gap:12px}
header h1{font-size:1.6rem;font-weight:300;letter-spacing:3px;
          background:linear-gradient(90deg,#4a9eff,#00c853);
          -webkit-background-clip:text;-webkit-text-fill-color:transparent}
.updated{font-size:.8rem;color:#666;margin-top:4px}

/* ── Stage legend ───────────────────────────────────────────── */
.legend{display:flex;gap:8px;flex-wrap:wrap}
.legend-item{padding:3px 12px;border-radius:20px;font-size:.75rem;
             background:color-mix(in srgb,var(--c) 18%,transparent);
             color:var(--c);border:1px solid color-mix(in srgb,var(--c) 40%,transparent)}

/* ── Country cards ──────────────────────────────────────────── */
.cards-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;
            margin-bottom:14px}
.card{background:#16213e;border-radius:12px;padding:22px 20px;
      border:1px solid #2a2a4a}
.card-country{font-size:1.5rem;font-weight:700;letter-spacing:5px;
              color:#c0c0c0;margin-bottom:12px}
.stage-badge{display:inline-block;padding:5px 18px;border-radius:20px;
             font-size:1.05rem;font-weight:600;margin-bottom:14px}
.level-track{background:#1a1a2e;border-radius:4px;height:5px;
             margin-bottom:14px;overflow:hidden}
.level-fill{height:100%;border-radius:4px;transition:width .3s}
.card-stats{display:grid;grid-template-columns:repeat(3,1fr);gap:4px}
.stat-item{text-align:center}
.stat-label{display:block;font-size:.65rem;color:#666;
            text-transform:uppercase;letter-spacing:1px}
.stat-value{display:block;font-size:.95rem;font-weight:600;margin-top:2px}
.card-date{text-align:right;font-size:.65rem;color:#444;margin-top:12px}
.card-error{display:flex;flex-direction:column;align-items:center;
            justify-content:center;min-height:140px}
.card-error-msg{color:#666;font-size:.85rem;margin-top:8px}

/* ── CN Signal bar ──────────────────────────────────────────── */
.cn-bar{background:#16213e;border-radius:10px;padding:13px 20px;
        border:1px solid #2a2a4a;margin-bottom:14px;
        display:flex;align-items:center;gap:14px;flex-wrap:wrap}
.cn-label{font-size:.7rem;text-transform:uppercase;letter-spacing:2px;
          color:#888;white-space:nowrap}
.cn-value{font-size:.95rem;font-weight:600}
.cn-date{font-size:.75rem;color:#555}
.cn-note{font-size:.7rem;color:#444;margin-left:auto}

/* ── Section titles ─────────────────────────────────────────── */
.section-title{font-size:.7rem;text-transform:uppercase;letter-spacing:2px;
               color:#555;margin-bottom:10px}

/* ── Chart grid (3 cols) ────────────────────────────────────── */
.charts-section{margin-bottom:14px}
.charts-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}

/* ── Heatmap grid (1:2) ─────────────────────────────────────── */
.heatmaps-section{margin-bottom:14px}
.heatmaps-grid{display:grid;grid-template-columns:1fr 2fr;gap:12px}

/* ── Image cards ────────────────────────────────────────────── */
.img-card{background:#16213e;border-radius:12px;overflow:hidden;
          border:1px solid #2a2a4a}
.img-label{padding:8px 14px;font-size:.65rem;text-transform:uppercase;
           letter-spacing:2px;color:#666;border-bottom:1px solid #1a1a2e}
.img-card img{width:100%;display:block}
.img-missing{padding:32px;text-align:center;color:#333;font-size:.8rem}

/* ── Footer ─────────────────────────────────────────────────── */
footer{text-align:center;margin-top:28px;padding:16px 0;
       font-size:.7rem;color:#333;border-top:1px solid #2a2a4a}

/* ── Responsive ─────────────────────────────────────────────── */
@media(max-width:900px){
  .cards-grid{grid-template-columns:1fr}
  .charts-grid{grid-template-columns:1fr}
  .heatmaps-grid{grid-template-columns:1fr}
}
"""


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

_HTML_TMPL = """\
<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Business Cycle Dashboard</title>
<style>
{css}
</style>
</head>
<body>
<div class="container">

<header>
  <div>
    <h1>Business Cycle Dashboard</h1>
    <p class="updated">Last updated: {updated}</p>
  </div>
  <div class="legend">
    <span class="legend-item" style="--c:#4a9eff">回復</span>
    <span class="legend-item" style="--c:#00c853">上昇</span>
    <span class="legend-item" style="--c:#ffab00">成熟</span>
    <span class="legend-item" style="--c:#ff6d00">軟化</span>
    <span class="legend-item" style="--c:#f44336">下降</span>
  </div>
</header>

<section>
  <div class="cards-grid">
{cards}
  </div>
  <div class="cn-bar">{cn}</div>
</section>

<section class="charts-section">
  <div class="section-title">Cycle Level &amp; Momentum</div>
  <div class="charts-grid">
{chart_cards}
  </div>
</section>

<section class="heatmaps-section">
  <div class="section-title">Stage &times; Factor / Sector Returns</div>
  <div class="heatmaps-grid">
{heatmap_cards}
  </div>
</section>

</div>
<footer>
  Data sources: e-Stat (JP) &middot; FRED (US/KR/CN) &middot; OECD SDMX &middot;
  Ken French Data Library &middot;
  Generated by <a href="https://github.com/kfuru1984/bcycle">bcycle-jp</a>
</footer>
</body>
</html>
"""


def generate() -> Path:
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # ── Country data ──────────────────────────────────────────────────────────
    countries = {c: _load_country(c) for c in COUNTRIES}

    cards_html = "\n".join(
        "    " + line
        for c in COUNTRIES
        for line in _country_card(countries[c]).splitlines()
    )

    # ── CN Signal ─────────────────────────────────────────────────────────────
    cn_html = _cn_signal_html()

    # ── Stage timeline charts ─────────────────────────────────────────────────
    chart_uris = {
        c: _b64(ROOT / "data" / c / "stage_timeline.png") for c in COUNTRIES
    }
    chart_cards_html = "\n".join(
        "    " + line
        for c in COUNTRIES
        for line in _img_card(
            c.upper() + " — Stage Timeline",
            chart_uris[c],
        ).splitlines()
    )

    # ── Heatmaps ──────────────────────────────────────────────────────────────
    heatmap_specs = [
        ("JP", ROOT / "data" / "jp" / "stage_factor_heatmap.png",
         "JP — Stage × Factor Returns"),
        ("US", ROOT / "data" / "us" / "stage_sector_heatmap.png",
         "US — Stage × Sector Returns (12 Industry)"),
    ]
    heatmap_cards_html = "\n".join(
        "    " + line
        for _, path, label in heatmap_specs
        for line in _img_card(label, _b64(path)).splitlines()
    )

    # ── Latest report link ────────────────────────────────────────────────────
    report_files = sorted((ROOT / "reports").glob("20??-??.md"), reverse=True)
    latest_report = report_files[0].stem if report_files else None
    if latest_report:
        now_str += f" &nbsp;·&nbsp; <a href='../reports/{latest_report}.md'>Report {latest_report}</a>"

    html = _HTML_TMPL.format(
        css=_CSS,
        updated=now_str,
        cards=cards_html,
        cn=cn_html,
        chart_cards=chart_cards_html,
        heatmap_cards=heatmap_cards_html,
    )

    out = DOCS_DIR / "index.html"
    out.write_text(html, encoding="utf-8")
    print(f"Dashboard: {out}  ({len(html):,} chars)")
    return out


if __name__ == "__main__":
    generate()
