"""
横断ダッシュボード生成スクリプト (Phase C/D 全面刷新版)。

読み込み:
  data/{jp,us,kr}/cycle_detail.json   ← run_cycle.py が出力
  data/{jp,us,kr}/stage_timeline.png
  config/stage_content.yaml

出力: docs/index.html (自己完結 HTML、画像は base64 埋め込み)

実行:
    python scripts/generate_dashboard.py
"""
from __future__ import annotations

import base64
import json
import os
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

ROOT     = Path(__file__).resolve().parents[1]
DOCS_DIR = ROOT / "docs"
DOCS_DIR.mkdir(exist_ok=True)

COUNTRIES = ["jp", "us", "kr"]

STAGE_COLORS: dict[str, str] = {
    "回復": "#3b82f6",
    "上昇": "#22c55e",
    "成熟": "#f59e0b",
    "軟化": "#f97316",
    "下降": "#ef4444",
}

CATEGORY_JA: dict[str, str] = {
    "production": "生産",
    "labor":      "労働",
    "housing":    "住宅",
    "consumption":"消費",
    "external":   "外需",
    "inflation":  "物価",
    "financial":  "金融",
    "investment": "設備投資",
    "sentiment":  "信頼感",
    "other":      "その他",
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _b64(path: Path) -> str | None:
    if not path.exists():
        return None
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def _load_detail(country: str) -> dict:
    p = ROOT / "data" / country / "cycle_detail.json"
    if not p.exists():
        return {"ok": False, "country": country.upper()}
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _load_stage_content() -> dict:
    p = ROOT / "config" / "stage_content.yaml"
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _fmt_value(value: float | None, ind_id: str, transform: str) -> str:
    if value is None:
        return "—"
    pct_transforms = ("yoy_pct", "mom_pct", "derived")
    if transform in pct_transforms or ind_id.endswith("_yoy"):
        return f"{value:+.2f}%"
    return f"{value:.2f}"


def _cn_signal_html() -> str:
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
                return "CN Signal: FRED_API_KEY 未設定"
            s   = fred.fetch({"series_id": "CHNBSCICP02STSAM"}, start=date(2000, 1, 1)).dropna()
            val = float(s.iloc[-1])
            dts = str(s.index[-1])[:7]
        color = "#22c55e" if val >= 0.0 else "#ef4444"
        label = "Expanding" if val >= 0.0 else "Contracting"
        return (
            f'<span class="cn-lbl">CN Signal</span>'
            f'<span class="cn-val" style="color:{color}">BCI {val:+.2f} ({label})</span>'
            f'<span class="cn-date">[{dts}]</span>'
            f'<span class="cn-note">OECD Mfg Business Confidence</span>'
        )
    except Exception as exc:
        return f'<span class="cn-lbl">CN Signal</span><span style="color:#666">取得失敗: {exc}</span>'


# ─────────────────────────────────────────────────────────────────────────────
# LLM Narrative
# ─────────────────────────────────────────────────────────────────────────────

def _predict_next_stage(detail: dict) -> str:
    stage    = detail.get("stage", "")
    level    = detail.get("level", 50.0)
    momentum = detail.get("momentum", 0.0)
    near_high = level > 58.0
    near_low  = level < 35.0

    if stage == "下降":
        return "下降継続（回復の兆し）" if momentum > 0.3 else "下降継続"
    if stage == "回復":
        return "回復局面継続（上昇移行準備）" if momentum > 0.3 else "回復継続"
    if stage == "上昇":
        return "上昇継続だが成熟入りの兆し" if (near_high or momentum < 0.1) else "上昇局面継続"
    if stage == "成熟":
        return "過熱継続（軟化前夜）" if momentum < -0.1 else "過熱継続"
    if stage == "軟化":
        return "軟化から下降へ移行の可能性" if momentum < -0.3 else "軟化継続"
    return stage + "継続"


def _generate_narrative(detail: dict) -> str:
    """2-3 sentence stage background narrative. Falls back to template on error."""
    try:
        import anthropic as _anthropic
        api_key = os.environ.get("ANTHROPIC_API_KEY") or ""
        if not api_key:
            raise ValueError("no key")

        top_lv_text = "\n".join(
            f"  - {f['name_ja']}: L{f['percentile']:.0f} {f['z']:+.1f}σ"
            for f in detail.get("top3_level", []) if f
        ) or "  (データなし)"
        top_mm_text = "\n".join(
            f"  - {f['name_ja']}: L{f['percentile']:.0f} {f['z']:+.1f}σ"
            for f in detail.get("top3_momentum", []) if f
        ) or "  (データなし)"
        rpr = detail.get("real_policy_rate")
        yc  = detail.get("yc_10y2y")
        rw  = detail.get("real_wage_yoy")

        prompt = f"""あなたはマクロ経済アナリストです。以下のデータから景気循環モデルのステージ判定背景を日本語で2-3文で記述してください。専門的かつ簡潔に、数値を交えて説明してください。

国: {detail['country']}
現在ステージ: {detail['stage']}
サイクルレベル: {detail['level']:.1f}/100
モメンタム: {detail['momentum']:+.2f}
中銀スタンス: {detail.get('cb_stance', '不明')}
実質政策金利: {f"{rpr:+.2f}%" if rpr is not None else "N/A"}
イールドカーブ: {f"{yc:+.2f}%" if yc is not None else "N/A"}
実質賃金: {f"{rw:+.2f}%" if rw is not None else "N/A"}

主導要因(高水準):
{top_lv_text}

モメンタム主導(変化最大):
{top_mm_text}

出力は純粋なテキストのみ(マークダウン不要)。"""

        client = _anthropic.Anthropic(api_key=api_key)
        resp = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text.strip()
    except Exception:
        lvl  = detail.get("level", 0)
        mom  = detail.get("momentum", 0)
        lbl  = "低水準" if lvl < 40 else ("高水準" if lvl >= 65 else "中位")
        mlbl = "上昇" if mom > 0 else "低下"
        weak = ", ".join(
            f["name_ja"] for f in detail.get("top3_momentum", [])[:2] if f
        ) or "—"
        rw = detail.get("real_wage_yoy")
        rw_str = (f"実質賃金{rw:+.2f}%で消費下支え" if rw and rw > 0 else
                  f"実質賃金{rw:+.2f}%で購買力圧迫" if rw and rw < 0 else "")
        cb = detail.get("cb_stance", "")
        return (
            f"全体レベル{lvl:.1f}({lbl})でモメンタム{mom:+.2f}と{mlbl}傾向。"
            f"弱い指標: {weak}。"
            + (f" 関連シグナル: {rw_str}。" if rw_str else "")
            + (f" 中銀スタンス: {cb}。" if cb else "")
        )


def _generate_outlook(detail: dict, next_stage: str) -> str:
    try:
        import anthropic as _anthropic
        api_key = os.environ.get("ANTHROPIC_API_KEY") or ""
        if not api_key:
            raise ValueError("no key")
        prompt = (
            f"現状は{detail['stage']} (level {detail['level']:.1f}, mom {detail['momentum']:+.2f})、"
            f"向こう3-6ヶ月の方向性を1文で記述。**{next_stage}**の形式で次のステージを太字で含めること。"
        )
        client = _anthropic.Anthropic(api_key=api_key)
        resp = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=150,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text.strip()
    except Exception:
        return (
            f"現状は{detail['stage']} (level {detail['level']:.1f}, mom {detail['momentum']:+.2f})、"
            f"向こう3-6ヶ月の方向性は**{next_stage}**。"
        )


def _compute_outlook_metrics(detail: dict) -> dict:
    mom  = detail.get("momentum", 0.0)
    rpr  = detail.get("real_policy_rate")
    rw   = detail.get("real_wage_yoy")
    cpi  = detail.get("core_cpi_yoy")
    stage = detail.get("stage", "")

    if mom > 0.5:
        growth = ("✅ 拡大加速",    "#22c55e")
    elif mom > 0:
        growth = ("↗ 緩やかな拡大", "#22c55e")
    elif mom > -0.5:
        growth = ("↘ 減速",         "#f97316")
    else:
        growth = ("↘↘ 急減速",      "#ef4444")

    if cpi is None:
        inflation = ("─ データなし", "#888888")
    elif cpi > 4.0:
        inflation = ("🔥 加速継続",  "#ef4444")
    elif cpi > 2.5:
        inflation = ("↑ 上昇中",    "#f97316")
    elif cpi > 0.5:
        inflation = ("─ 中立",      "#888888")
    else:
        inflation = ("↓ 低インフレ", "#3b82f6")

    if rpr is None:
        rate = ("─ データなし",       "#888888")
    elif rpr > 1.0:
        rate = ("↑↑ 利上げ加速織込", "#ef4444")
    elif rpr > 0:
        rate = ("↑ 利上げ余地",      "#f97316")
    elif rpr > -1.0:
        rate = ("─ 据え置き優勢",     "#888888")
    else:
        rate = ("↓ 利下げ転換の可能性", "#22c55e")

    labor_map = {
        "回復": ("↗ 改善中",   "#22c55e"),
        "上昇": ("🔥 タイト",  "#ef4444"),
        "成熟": ("🔥 タイト",  "#ef4444"),
        "軟化": ("↘ 緩和傾向", "#f97316"),
        "下降": ("↘ 緩和傾向", "#f97316"),
    }
    labor = labor_map.get(stage, ("─ 中立", "#888888"))

    if rw is None:
        wage = ("─ データなし",  "#888888")
    elif rw > 0.5:
        wage = ("↑ 実質プラス", "#22c55e")
    elif rw > -0.5:
        wage = ("─ 実質ほぼゼロ", "#888888")
    else:
        wage = ("↓ 実質マイナス", "#ef4444")

    return {
        "成長":     growth,
        "インフレ": inflation,
        "政策金利": rate,
        "労働市場": labor,
        "実質賃金": wage,
    }


def _compute_risks(detail: dict) -> list[dict]:
    risks = []
    rpr   = detail.get("real_policy_rate")
    yc    = detail.get("yc_10y2y")
    cpi   = detail.get("core_cpi_yoy")
    stage = detail.get("stage", "")

    if rpr is not None and rpr < -1.0:
        risks.append({
            "level": "high",
            "title": "ビハインド・ザ・カーブ",
            "desc":  f"政策金利がTaylor Rule比 → 利上げ遅れによるリフレ加速懸念",
        })
    if yc is not None and yc < 0.3 and stage in ("成熟", "上昇"):
        risks.append({
            "level": "medium",
            "title": "カーブフラット化",
            "desc":  f"2s10s {yc:+.0f}bps — 利上げ後期/景気減速売兆",
        })
    if cpi is not None and cpi > 4.5:
        risks.append({
            "level": "medium",
            "title": "インフレ加速",
            "desc":  f"コアCPI {cpi:.2f}% — 高インフレ継続",
        })
    if rpr is not None and -1.0 <= rpr < 0.0 and stage in ("上昇", "成熟"):
        risks.append({
            "level": "low",
            "title": "やや緩和的スタンス",
            "desc":  f"実質金利 {rpr:+.1f}% — 景気過熱リスクに注意",
        })
    return risks


# ─────────────────────────────────────────────────────────────────────────────
# HTML Component Builders
# ─────────────────────────────────────────────────────────────────────────────

def _header_card(detail: dict, sc: dict) -> str:
    stage  = detail.get("stage", "-")
    level  = detail.get("level", 0.0)
    mom    = detail.get("momentum", 0.0)
    color  = STAGE_COLORS.get(stage, "#888888")
    sc_cfg = sc["stages"].get(stage, {})
    trend  = sc_cfg.get("trend_label", "")
    biz    = sc_cfg.get("biz_cycle", "")
    c_cfg  = sc["countries"].get(detail.get("country", "").lower(), {})
    name_ja = c_cfg.get("name_ja", detail.get("country", ""))
    mom_color = "#22c55e" if mom > 0.05 else ("#ef4444" if mom < -0.05 else "#888888")

    stage_dots = ""
    for s in ["回復", "上昇", "成熟", "軟化", "下降"]:
        sc_col = STAGE_COLORS[s]
        active = "active" if s == stage else ""
        stage_dots += (
            f'<span class="stage-dot {active}" style="'
            f'background:{sc_col}22;color:{sc_col};border:1px solid {sc_col}55'
            f'{";background:" + sc_col + ";color:#0f172a" if active else ""}'
            f'">{s}</span>'
        )

    return f"""\
<div class="header-card" style="border-left:4px solid {color}">
  <div class="hc-left">
    <div class="hc-title">
      <span class="country-name">{name_ja}</span>
      <span class="country-code">{detail.get('country','')}</span>
    </div>
    <div class="hc-stage-row">
      <span class="stage-badge" style="background:{color};color:#0f172a">{stage}</span>
      <span class="stage-trend">{trend}</span>
    </div>
    <div class="stage-dots">{stage_dots}</div>
  </div>
  <div class="hc-right">
    <div class="hc-metric">
      <div class="hm-label">サイクルレベル</div>
      <div class="hm-value">{level:.1f}</div>
      <div class="hm-unit">/100</div>
    </div>
    <div class="hc-metric">
      <div class="hm-label">モメンタム</div>
      <div class="hm-value" style="color:{mom_color}">{mom:+.2f}</div>
      <div class="hm-unit">z-score</div>
    </div>
    <div class="hc-metric">
      <div class="hm-label">ビジネスサイクル</div>
      <div class="hm-value hm-cycle">{biz}</div>
      <div class="hm-unit">{detail.get('as_of','')}</div>
    </div>
  </div>
</div>"""


def _exposure_html(detail: dict, sc: dict) -> str:
    stage  = detail.get("stage", "-")
    sc_cfg = sc["stages"].get(stage, {})
    color  = sc_cfg.get("exposure_color", "#888888")
    exp    = sc_cfg.get("exposure", "")
    desc   = sc_cfg.get("exposure_desc", "").strip()
    return f"""\
<div class="info-card">
  <div class="ic-title">📊 推奨エクスポージャー</div>
  <div class="exposure-bar" style="background:{color}">
    <span class="exp-text">{exp}</span>
  </div>
  <p class="ic-desc">{desc}</p>
</div>"""


def _sector_factor_html(detail: dict, sc: dict) -> str:
    stage  = detail.get("stage", "-")
    sc_cfg = sc["stages"].get(stage, {})
    buy    = sc_cfg.get("sector_buy", [])
    sell   = sc_cfg.get("sector_sell", [])
    factors= sc_cfg.get("factors", [])
    bc     = sc_cfg.get("sector_buy_color",  "#22c55e")
    sc_col = sc_cfg.get("sector_sell_color", "#ef4444")

    buy_html  = "・".join(f'<span style="color:{bc}">{s}</span>' for s in buy)
    sell_html = "・".join(f'<span style="color:{sc_col}">{s}</span>' for s in sell)
    fac_html  = "・".join(factors)

    return f"""\
<div class="info-card">
  <div class="ic-title">📋 セクター・ファクター</div>
  <div class="sf-row"><span class="sf-label">買い</span><span class="sf-items">{buy_html}</span></div>
  <div class="sf-row"><span class="sf-label sell">売り</span><span class="sf-items">{sell_html}</span></div>
  <div class="sf-row"><span class="sf-label fac">ファクター</span><span class="sf-items">{fac_html}</span></div>
</div>"""


def _watch_points_html(detail: dict, sc: dict) -> str:
    country = detail.get("country", "").lower()
    stage   = detail.get("stage", "-")
    pts = sc.get("countries", {}).get(country, {}).get("watch_points", {}).get(stage, [])
    items = "".join(f"<li>{p}</li>" for p in pts) if pts else "<li>データなし</li>"
    return f"""\
<div class="info-card">
  <div class="ic-title">🎯 重要監視ポイント</div>
  <ul class="watch-list">{items}</ul>
</div>"""


def _factor_row_html(f: dict) -> str:
    if not f:
        return ""
    cat_ja = CATEGORY_JA.get(f.get("category", "other"), "その他")
    name   = f.get("name_ja", f.get("id", ""))
    pct    = f.get("percentile")
    z      = f.get("z")
    val    = f.get("value")
    transform = f.get("transform", "level")
    ind_id    = f.get("id", "")
    pct_str = f"L{pct:.0f}" if pct is not None else "L—"
    z_str   = f"{z:+.1f}σ"  if z   is not None else "—σ"
    val_str = _fmt_value(val, ind_id, transform)
    return (
        f'<div class="factor-row">'
        f'<span class="cat-badge cat-{f.get("category","other")}">{cat_ja}</span>'
        f'<span class="factor-name">{name}</span>'
        f'<span class="factor-stats">{pct_str} {z_str} {val_str}</span>'
        f'</div>'
    )


def _background_html(detail: dict, narrative: str) -> str:
    stage   = detail.get("stage", "-")
    color   = STAGE_COLORS.get(stage, "#888888")
    cb      = detail.get("cb_stance", "")
    cb_dtl  = detail.get("cb_detail", "")
    rw      = detail.get("real_wage_yoy")
    rw_str  = f" / 実質賃金 {rw:+.2f}%" if rw is not None else ""

    right_info = ""
    parts = []
    if cb:
        parts.append(f"中銀スタンス: {cb}")
    if cb_dtl:
        parts.append(cb_dtl)
    if rw_str:
        parts.append(rw_str.strip(" /"))
    right_info = " / ".join(parts)

    top3_lv = detail.get("top3_level",    [])
    top3_mm = detail.get("top3_momentum", [])

    lv_rows = "".join(_factor_row_html(f) for f in top3_lv)
    mm_rows = "".join(_factor_row_html(f) for f in top3_mm)

    # Bold the **...** in narrative (from LLM)
    import re
    narrative_html = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", narrative)

    return f"""\
<div class="bg-section">
  <div class="bg-header">
    <span class="bg-title">🔍 ステージ判定の背景: <span style="color:{color}">{stage}</span></span>
    <span class="bg-right">{right_info}</span>
  </div>
  <p class="bg-narrative">{narrative_html}</p>
  <div class="factor-grid">
    <div class="factor-col">
      <div class="fc-header">主導要因 <span class="fc-sub">(高水準/ピーク圏)</span></div>
      {lv_rows or '<div class="factor-row"><span style="color:#555">データなし</span></div>'}
    </div>
    <div class="factor-col">
      <div class="fc-header">モメンタム主導 <span class="fc-sub">(z絶対値最大)</span></div>
      {mm_rows or '<div class="factor-row"><span style="color:#555">データなし</span></div>'}
    </div>
  </div>
</div>"""


def _outlook_html(detail: dict, outlook_text: str, next_stage: str,
                  metrics: dict, risks: list[dict]) -> str:
    import re
    outlook_html = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", outlook_text)

    metric_html = ""
    for label, (text, color) in metrics.items():
        metric_html += (
            f'<div class="metric-cell">'
            f'<div class="mc-label">{label}</div>'
            f'<div class="mc-value" style="color:{color}">{text}</div>'
            f'</div>'
        )

    risk_html = ""
    if risks:
        risk_html = '<div class="risk-header">⚠ リスク要因</div>'
        for r in risks:
            lvl_color = {"high": "#ef4444", "medium": "#f97316", "low": "#f59e0b"}.get(r["level"], "#888")
            risk_html += (
                f'<div class="risk-item">'
                f'<span class="risk-dot" style="background:{lvl_color}"></span>'
                f'<div><div class="risk-title">{r["title"]}</div>'
                f'<div class="risk-desc">{r["desc"]}</div></div>'
                f'</div>'
            )

    return f"""\
<div class="outlook-section">
  <div class="outlook-header">
    <span class="ols-title">🔮 今後の見通し (3-6ヶ月)</span>
    <span class="ols-conf">{"確度高" if detail.get("momentum", 0) > 0.3 or detail.get("momentum", 0) < -0.3 else "確度中"}</span>
  </div>
  <div class="ols-next">次に予想される展開: <strong style="color:#f59e0b">{next_stage}</strong></div>
  <p class="ols-text">{outlook_html}</p>
  <div class="metrics-row">{metric_html}</div>
  {risk_html}
</div>"""


# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────

_CSS = """\
*{box-sizing:border-box;margin:0;padding:0}
body{background:#0f172a;color:#cbd5e1;font-family:'Segoe UI',system-ui,sans-serif;
     line-height:1.5;padding:16px 20px}
strong{color:#f1f5f9}
ul{list-style:none}
a{color:#60a5fa}

/* ── Top header ─────────────────────────────────────────────── */
.page-header{display:flex;align-items:flex-start;justify-content:space-between;
             padding:12px 0 10px;flex-wrap:wrap;gap:8px}
.page-header h1{font-size:1.3rem;font-weight:300;letter-spacing:3px;
  background:linear-gradient(90deg,#60a5fa,#34d399);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent}
.updated{font-size:.7rem;color:#475569;margin-top:3px}

/* ── CN bar ─────────────────────────────────────────────────── */
.cn-bar{background:#1e293b;border-radius:8px;padding:8px 16px;
        border:1px solid #334155;margin-bottom:10px;
        display:flex;align-items:center;gap:12px;flex-wrap:wrap}
.cn-lbl{font-size:.65rem;text-transform:uppercase;letter-spacing:2px;color:#64748b}
.cn-val{font-size:.9rem;font-weight:600}
.cn-date{font-size:.7rem;color:#475569}
.cn-note{font-size:.65rem;color:#334155;margin-left:auto}

/* ── Country tabs ───────────────────────────────────────────── */
.tabs{display:flex;gap:6px;margin-bottom:12px}
.tab-btn{padding:8px 24px;border-radius:8px;border:1px solid #334155;
         background:#1e293b;color:#94a3b8;font-size:.9rem;cursor:pointer;
         transition:all .15s}
.tab-btn:hover{background:#2d3f55;color:#e2e8f0}
.tab-btn.active{border-color:#60a5fa;background:#1e3a5f;color:#93c5fd;font-weight:600}

/* ── Panel show/hide ─────────────────────────────────────────── */
.panel{display:none}
.panel.active{display:block}

/* ── Header card ─────────────────────────────────────────────── */
.header-card{background:#1e293b;border-radius:12px;padding:20px 24px;
             border:1px solid #334155;margin-bottom:10px;
             display:flex;justify-content:space-between;align-items:flex-start;gap:16px;
             flex-wrap:wrap}
.hc-left{flex:1;min-width:220px}
.hc-title{display:flex;align-items:baseline;gap:10px;margin-bottom:10px}
.country-name{font-size:1.5rem;font-weight:700;color:#f1f5f9}
.country-code{font-size:.8rem;color:#64748b;letter-spacing:2px}
.hc-stage-row{display:flex;align-items:center;gap:12px;margin-bottom:12px}
.stage-badge{padding:4px 14px;border-radius:20px;font-size:1.0rem;font-weight:700}
.stage-trend{color:#94a3b8;font-size:.85rem}
.stage-dots{display:flex;gap:6px;flex-wrap:wrap}
.stage-dot{padding:3px 10px;border-radius:16px;font-size:.75rem;cursor:default}
.hc-right{display:flex;gap:20px;flex-wrap:wrap}
.hc-metric{text-align:center;min-width:90px}
.hm-label{font-size:.6rem;text-transform:uppercase;letter-spacing:1px;color:#64748b;margin-bottom:4px}
.hm-value{font-size:1.7rem;font-weight:700;color:#f1f5f9;line-height:1}
.hm-cycle{font-size:1.0rem!important}
.hm-unit{font-size:.65rem;color:#475569;margin-top:2px}

/* ── 3-col info cards ───────────────────────────────────────── */
.info-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-bottom:10px}
.info-card{background:#1e293b;border-radius:10px;padding:16px;border:1px solid #334155}
.ic-title{font-size:.7rem;text-transform:uppercase;letter-spacing:1.5px;
          color:#64748b;margin-bottom:10px}
.exposure-bar{border-radius:6px;padding:8px 14px;margin-bottom:8px;text-align:center}
.exp-text{font-size:.85rem;font-weight:600;color:#0f172a}
.ic-desc{font-size:.78rem;color:#94a3b8;line-height:1.5}
.sf-row{display:flex;gap:8px;margin-bottom:6px;align-items:flex-start}
.sf-label{font-size:.7rem;color:#64748b;min-width:60px;padding-top:2px}
.sf-label.sell{color:#f87171}
.sf-label.fac{color:#93c5fd}
.sf-items{font-size:.78rem;line-height:1.6}
.watch-list li{font-size:.78rem;color:#94a3b8;padding:3px 0;
               border-bottom:1px solid #1e293b}
.watch-list li::before{content:"· ";color:#475569}

/* ── Background section ─────────────────────────────────────── */
.bg-section{background:#1e293b;border-radius:10px;padding:16px 20px;
            border:1px solid #334155;margin-bottom:10px}
.bg-header{display:flex;justify-content:space-between;align-items:baseline;
           flex-wrap:wrap;gap:8px;margin-bottom:8px}
.bg-title{font-size:.85rem;font-weight:600;color:#e2e8f0}
.bg-right{font-size:.7rem;color:#64748b}
.bg-narrative{font-size:.82rem;color:#94a3b8;line-height:1.7;
              background:#0f172a;border-radius:6px;padding:10px 14px;margin-bottom:12px}
.factor-grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.factor-col{background:#0f172a;border-radius:8px;padding:12px}
.fc-header{font-size:.65rem;text-transform:uppercase;letter-spacing:1px;
           color:#475569;margin-bottom:8px}
.fc-sub{font-size:.6rem;color:#334155}
.factor-row{display:flex;align-items:center;gap:8px;padding:5px 0;
            border-bottom:1px solid #1a2535}
.factor-row:last-child{border-bottom:none}
.cat-badge{font-size:.6rem;padding:2px 6px;border-radius:4px;white-space:nowrap;
           background:#1e293b;color:#94a3b8;min-width:36px;text-align:center}
.cat-production{background:#1e3a5f22;color:#60a5fa}
.cat-labor{background:#1a3a2622;color:#4ade80}
.cat-housing{background:#2d2a1022;color:#fbbf24}
.cat-consumption{background:#2d1a2a22;color:#c084fc}
.cat-external{background:#1a2a3a22;color:#38bdf8}
.cat-inflation{background:#3a1a1a22;color:#f87171}
.cat-financial{background:#1a2d3a22;color:#67e8f9}
.cat-sentiment{background:#2a1a3022;color:#a78bfa}
.factor-name{flex:1;font-size:.78rem;color:#cbd5e1}
.factor-stats{font-size:.72rem;color:#64748b;white-space:nowrap;text-align:right}

/* ── Outlook section ─────────────────────────────────────────── */
.outlook-section{background:#1e2d45;border-radius:10px;padding:16px 20px;
                 border:1px solid #2d4a6e;margin-bottom:10px}
.outlook-header{display:flex;justify-content:space-between;align-items:center;
                margin-bottom:6px}
.ols-title{font-size:.85rem;font-weight:600;color:#e2e8f0}
.ols-conf{font-size:.65rem;padding:2px 8px;border-radius:10px;
          background:#f59e0b22;color:#fbbf24;border:1px solid #f59e0b44}
.ols-next{font-size:.82rem;color:#94a3b8;margin-bottom:6px}
.ols-text{font-size:.82rem;color:#94a3b8;line-height:1.7;
          background:#0f172a;border-radius:6px;padding:10px 14px;margin-bottom:12px}
.metrics-row{display:grid;grid-template-columns:repeat(5,1fr);gap:8px;margin-bottom:10px}
.metric-cell{background:#0f172a;border-radius:6px;padding:10px 8px;text-align:center}
.mc-label{font-size:.6rem;color:#475569;text-transform:uppercase;
          letter-spacing:1px;margin-bottom:4px}
.mc-value{font-size:.78rem;font-weight:600}
.risk-header{font-size:.7rem;text-transform:uppercase;letter-spacing:1px;
             color:#f59e0b;margin-bottom:6px}
.risk-item{display:flex;gap:10px;align-items:flex-start;
           background:#0f172a;border-radius:6px;padding:8px 12px;margin-bottom:6px}
.risk-dot{width:10px;height:10px;border-radius:50%;flex-shrink:0;margin-top:4px}
.risk-title{font-size:.8rem;font-weight:600;color:#f1f5f9}
.risk-desc{font-size:.72rem;color:#64748b;margin-top:2px}

/* ── Chart ───────────────────────────────────────────────────── */
.chart-section{background:#1e293b;border-radius:10px;overflow:hidden;
               border:1px solid #334155;margin-bottom:10px}
.chart-label{padding:8px 14px;font-size:.65rem;text-transform:uppercase;
             letter-spacing:2px;color:#475569;border-bottom:1px solid #0f172a}
.chart-section img{width:100%;display:block}
.chart-missing{padding:32px;text-align:center;color:#334155;font-size:.8rem}

/* ── Footer ─────────────────────────────────────────────────── */
footer{text-align:center;margin-top:20px;padding:12px 0;
       font-size:.65rem;color:#334155;border-top:1px solid #1e293b}

/* ── Inventory cycle section ─────────────────────────────── */
.inv-section{background:#111827;border-radius:12px;padding:20px 24px;
             border:1px solid #334155;margin-top:20px}
.inv-section-header{font-size:.9rem;font-weight:600;letter-spacing:2px;
                    text-transform:uppercase;color:#94a3b8;margin-bottom:16px;
                    padding-bottom:8px;border-bottom:1px solid #1e293b}
.inv-global{background:#1e293b;border-radius:10px;overflow:hidden;
            border:1px solid #334155;margin-bottom:14px}
.inv-global img{width:100%;display:block}
.inv-ts-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:10px;margin-bottom:14px}
.inv-sector-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:10px}
.inv-chart{background:#1e293b;border-radius:10px;overflow:hidden;border:1px solid #334155}
.inv-chart-label{padding:6px 12px;font-size:.6rem;text-transform:uppercase;
                 letter-spacing:2px;color:#475569;border-bottom:1px solid #0f172a}
.inv-chart img{width:100%;display:block}
.inv-chart-missing{padding:24px;text-align:center;color:#334155;font-size:.78rem}

/* ── Responsive ─────────────────────────────────────────────── */
@media(max-width:900px){
  .info-grid{grid-template-columns:1fr}
  .factor-grid{grid-template-columns:1fr}
  .metrics-row{grid-template-columns:repeat(2,1fr)}
  .hc-right{justify-content:flex-start}
  .inv-ts-grid{grid-template-columns:1fr}
  .inv-sector-grid{grid-template-columns:1fr}
}
"""

_JS = """\
function showTab(country) {
  document.querySelectorAll('.panel').forEach(function(p){ p.classList.remove('active'); });
  document.querySelectorAll('.tab-btn').forEach(function(b){ b.classList.remove('active'); });
  document.getElementById('panel-' + country).classList.add('active');
  document.getElementById('tab-' + country).classList.add('active');
}
"""


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def _build_country_panel(country: str, detail: dict, sc: dict, chart_uri: str | None) -> str:
    if not detail.get("ok"):
        return f'<div class="panel" id="panel-{country}"><p style="padding:40px;color:#555">データ取得失敗</p></div>'

    next_stage = _predict_next_stage(detail)
    narrative  = _generate_narrative(detail)
    outlook_t  = _generate_outlook(detail, next_stage)
    metrics    = _compute_outlook_metrics(detail)
    risks      = _compute_risks(detail)

    chart_html = ""
    if chart_uri:
        chart_html = (
            f'<div class="chart-section">'
            f'<div class="chart-label">Cycle Level &amp; Momentum</div>'
            f'<img src="{chart_uri}" alt="{country.upper()} stage timeline">'
            f'</div>'
        )
    else:
        chart_html = (
            f'<div class="chart-section">'
            f'<div class="chart-label">Cycle Level &amp; Momentum</div>'
            f'<div class="chart-missing">チャート未生成</div>'
            f'</div>'
        )

    body = "\n".join([
        _header_card(detail, sc),
        '<div class="info-grid">',
        _exposure_html(detail, sc),
        _sector_factor_html(detail, sc),
        _watch_points_html(detail, sc),
        '</div>',
        _background_html(detail, narrative),
        _outlook_html(detail, outlook_t, next_stage, metrics, risks),
        chart_html,
    ])

    return f'<div class="panel active" id="panel-{country}">\n{body}\n</div>'


INV_COUNTRIES_TS     = ["jp", "us", "eu", "cn"]
INV_COUNTRIES_SECTOR = ["jp", "us"]
INV_LABELS           = {"jp": "日本", "us": "米国", "eu": "ユーロ圏", "cn": "中国"}


def _build_inv_cycle_section() -> str:
    """在庫循環分析セクションの HTML を構築する。"""

    def _chart(uri: str | None, label: str) -> str:
        inner = (
            f'<img src="{uri}" alt="{label}">'
            if uri
            else f'<div class="inv-chart-missing">{label}: チャート未生成</div>'
        )
        return (
            f'<div class="inv-chart">'
            f'<div class="inv-chart-label">{label}</div>'
            f'{inner}'
            f'</div>'
        )

    # Global activity comparison
    global_uri = _b64(ROOT / "data" / "inv_cycle_activity.png")
    if global_uri:
        global_html = (
            f'<div class="inv-global">'
            f'<img src="{global_uri}" alt="製造業活動指標比較">'
            f'</div>'
        )
    else:
        global_html = (
            '<div class="inv-global">'
            '<div class="inv-chart-missing">製造業活動指標比較: チャート未生成</div>'
            '</div>'
        )

    # Per-country timeseries (2x2 grid)
    ts_items = []
    for c in INV_COUNTRIES_TS:
        uri = _b64(ROOT / "data" / c / "inv_cycle_ts.png")
        ts_items.append(_chart(uri, INV_LABELS.get(c, c.upper())))
    ts_html = '<div class="inv-ts-grid">' + "\n".join(ts_items) + "</div>"

    # Per-country sector charts (JP, US only)
    sector_items = []
    for c in INV_COUNTRIES_SECTOR:
        uri = _b64(ROOT / "data" / c / "inv_cycle_sector.png")
        sector_items.append(_chart(uri, f"{INV_LABELS.get(c, c.upper())} — 業種別断面"))
    sector_html = '<div class="inv-sector-grid">' + "\n".join(sector_items) + "</div>"

    return (
        '<div class="inv-section">\n'
        '<div class="inv-section-header">在庫循環分析</div>\n'
        + global_html + "\n"
        + ts_html + "\n"
        + sector_html + "\n"
        + "</div>"
    )


def generate() -> Path:
    from dotenv import load_dotenv
    load_dotenv()

    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    sc = _load_stage_content()

    # ── Per-country panels ─────────────────────────────────────────────
    panels_html = ""
    first = True
    for c in COUNTRIES:
        detail    = _load_detail(c)
        chart_uri = _b64(ROOT / "data" / c / "stage_timeline.png")
        panel     = _build_country_panel(c, detail, sc, chart_uri)
        # Only first panel is active by default
        if not first:
            panel = panel.replace('class="panel active"', 'class="panel"', 1)
        panels_html += panel + "\n"
        first = False

    # ── Tab buttons ────────────────────────────────────────────────────
    tab_html = ""
    for i, c in enumerate(COUNTRIES):
        detail  = _load_detail(c)
        name_ja = sc.get("countries", {}).get(c, {}).get("name_ja", c.upper())
        stage   = detail.get("stage", "—")
        color   = STAGE_COLORS.get(stage, "#888888")
        active  = "active" if i == 0 else ""
        tab_html += (
            f'<button class="tab-btn {active}" id="tab-{c}" onclick="showTab(\'{c}\')">'
            f'{name_ja} '
            f'<span style="color:{color};font-size:.75rem">{stage}</span>'
            f'</button>\n'
        )

    # ── CN signal ──────────────────────────────────────────────────────
    cn_html = _cn_signal_html()

    # ── Inventory cycle section ────────────────────────────────────────
    inv_cycle_html = _build_inv_cycle_section()

    # ── Latest report link ─────────────────────────────────────────────
    report_files = sorted((ROOT / "reports").glob("20??-??.md"), reverse=True)
    if report_files:
        stem = report_files[0].stem
        now_str += f" &nbsp;·&nbsp; <a href='../reports/{stem}.md'>Report {stem}</a>"

    html = f"""\
<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Business Cycle Dashboard</title>
<style>
{_CSS}
</style>
</head>
<body>
<div class="container" style="max-width:1400px;margin:0 auto">

<div class="page-header">
  <div>
    <h1>Business Cycle Dashboard</h1>
    <p class="updated">Last updated: {now_str}</p>
  </div>
</div>

<div class="cn-bar">{cn_html}</div>

<div class="tabs">
{tab_html}</div>

{panels_html}

{inv_cycle_html}

</div>
<footer>
  Data sources: e-Stat (JP) &middot; FRED (US/KR/CN) &middot; OECD SDMX &middot;
  Ken French Data Library &middot;
  Generated by <a href="https://github.com/kfuru1984/bcycle">bcycle-jp</a>
</footer>
<script>
{_JS}
</script>
</body>
</html>
"""

    out = DOCS_DIR / "index.html"
    out.write_text(html, encoding="utf-8")
    print(f"Dashboard: {out}  ({len(html):,} chars)")
    return out


if __name__ == "__main__":
    generate()
