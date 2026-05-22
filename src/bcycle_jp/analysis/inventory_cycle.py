"""
在庫循環分析モジュール。

出荷在庫バランス = 出荷YoY% − 在庫YoY%
  正値 → 出荷増・在庫減 → 生産加速シグナル
  負値 → 出荷減・在庫増 → 調整フェーズ

4フェーズ (Mizuho 2026-05-14 定義):
  積み増し加速 : balance > 0 かつ前月比 > 0
  積み増し     : balance > 0 かつ前月比 < 0
  調整加速     : balance < 0 かつ前月比 > 0  (底打ちからの回復)
  調整         : balance < 0 かつ前月比 < 0

出力:
  data/inv_cycle_activity.png         — JP/US/EU/CN 製造業活動指標比較
  data/{country}/inv_cycle_ts.png     — IPI + 出荷在庫バランス 時系列
  data/{country}/inv_cycle_sector.png — セクター別スナップショット (JP/US)
"""
from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as _fm
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[3]

_JP_FONT_CANDIDATES = ["Noto Sans JP", "Yu Gothic", "Meiryo", "MS Gothic", "MS PGothic"]
_jp_font: str | None = None
_available_names = {_fm.FontProperties(fname=f).get_name()
                    for f in _fm.findSystemFonts(fontext="ttf")}
for _cand in _JP_FONT_CANDIDATES:
    if _cand in _available_names:
        _jp_font = _cand
        break

log = logging.getLogger(__name__)

_PHASE_COLORS = {
    "積み増し加速": "#27ae60",
    "積み増し":     "#a8d8a8",
    "調整加速":     "#e67e22",
    "調整":         "#e74c3c",
    "不明":         "#cccccc",
    "データなし":   "#cccccc",
}
_COUNTRY_COLORS = {"JP": "#e74c3c", "US": "#3498db", "EU": "#2ecc71", "CN": "#f39c12"}


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _set_jp_font() -> None:
    if _jp_font:
        plt.rcParams["font.family"] = _jp_font


def _compute_yoy(series: pd.Series) -> pd.Series:
    """YoY% from level index."""
    return (series / series.shift(12) - 1) * 100


def _compute_balance(shipments: pd.Series, inventories: pd.Series) -> pd.Series:
    """出荷在庫バランス = 出荷YoY% − 在庫YoY%."""
    ship_yoy = _compute_yoy(shipments)
    inv_yoy  = _compute_yoy(inventories)
    idx      = ship_yoy.index.intersection(inv_yoy.index)
    return (ship_yoy.loc[idx] - inv_yoy.loc[idx]).rename("balance")


def detect_phase(balance: pd.Series, window: int = 3) -> str:
    """Detect current inventory cycle phase."""
    clean = balance.dropna()
    if len(clean) < 2:
        return "不明"
    recent    = clean.tail(window)
    current   = float(recent.iloc[-1])
    direction = float(recent.diff().dropna().mean()) if len(recent) >= 2 else 0.0
    if current >= 0:
        return "積み増し加速" if direction > 0 else "積み増し"
    else:
        return "調整加速" if direction > 0 else "調整"


def _sector_stats(series: pd.Series, months_back: int = 12) -> dict[str, float] | None:
    """Peak, trough, Q-4 (12m ago), current over trailing 36-month window."""
    clean = series.dropna()
    if len(clean) < 13:
        return None
    window  = clean.iloc[-36:] if len(clean) >= 36 else clean
    peak    = float(window.max())
    trough  = float(window.min())
    current = float(clean.iloc[-1])
    q4      = float(clean.iloc[-months_back]) if len(clean) >= months_back else float("nan")
    return {"peak": peak, "trough": trough, "q4": q4, "current": current}


def find_peak_bottom(series: pd.Series, window: int = 36) -> tuple[float, float]:
    """Return (peak, bottom) over trailing window months."""
    clean = series.dropna()
    if clean.empty:
        return float("nan"), float("nan")
    w = clean.iloc[-window:] if len(clean) >= window else clean
    return float(w.max()), float(w.min())


def calc_recovery_score(balance: pd.Series) -> float:
    """
    ボトムからピークに向けて何合目まで回復しているかを0-100で返す。
    = (直近値 - 直近ボトム) / (直近ピーク - 直近ボトム) * 100
    - 100超え: ピークを更新中（積み増し加速）
    - 0未満: ボトムを更新中（調整加速）
    - ピークとボトムは find_peak_bottom() の結果を使う
    """
    clean = balance.dropna()
    if len(clean) < 2:
        return float("nan")
    peak, bottom = find_peak_bottom(clean)
    if np.isnan(peak) or np.isnan(bottom):
        return float("nan")
    denom = peak - bottom
    if denom == 0.0:
        return float("nan")
    return (float(clean.iloc[-1]) - bottom) / denom * 100.0


# ─────────────────────────────────────────────────────────────────────────────
# FRED helper
# ─────────────────────────────────────────────────────────────────────────────

def _load_fred(series_id: str, start: date) -> pd.Series:
    from bcycle_jp.adapters.fred import FredAdapter
    return FredAdapter().fetch({"series_id": series_id}, start=start).dropna()


# ─────────────────────────────────────────────────────────────────────────────
# Country loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_jp(start: date = date(2018, 1, 1)) -> dict[str, Any]:
    """JP: e-Stat METI shipments/inventories + FRED IPI."""
    from bcycle_jp.adapters.estat import EStatAdapter
    estat = EStatAdapter()
    result: dict[str, Any] = {"country": "JP", "ok": False}

    try:
        ipi_yoy  = _load_fred("JPNPRINTO01GYSAM", start=date(2000, 1, 1))
        activity = _load_fred("JPNLOLITONOSTSAM",  start=date(2000, 1, 1))

        def _estat(stats_id: str, cat01: str) -> pd.Series:
            return estat.fetch({
                "stats_data_id": stats_id,
                "time_from_cat": "time",
                "filters": {"cat01": cat01},
            }, start=start).dropna()

        ship_agg = _estat("0004015801", "0002000")
        inv_agg  = _estat("0004015802", "0002000")
        balance  = _compute_balance(ship_agg, inv_agg)

        cfg_path = ROOT / "config" / "jp" / "manufacturing.yaml"
        with open(cfg_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        sectors: dict[str, dict] = {}
        for sec in cfg.get("sectors", []):
            try:
                sh  = _estat("0004015801", sec["cat01"])
                iv  = _estat("0004015802", sec["cat01"])
                bal = _compute_balance(sh, iv)
                st  = _sector_stats(bal)
                if st is not None:
                    st["recovery_score"] = calc_recovery_score(bal)
                    st["cycle_stage"]    = sec.get("cycle_stage", "mid")
                    sectors[sec["name_ja"]] = st
            except Exception as exc:
                log.warning("JP sector %s skipped: %s", sec["name_ja"], exc)

        result.update({
            "ok": True,
            "ipi_yoy": ipi_yoy,
            "balance": balance,
            "phase":   detect_phase(balance),
            "sectors": sectors,
            "activity": activity,
            "activity_label": "JP CLI (OECD)",
        })
    except Exception as exc:
        log.error("JP load failed: %s", exc)
        result["error"] = str(exc)
    return result


def load_us(start: date = date(2000, 1, 1)) -> dict[str, Any]:
    """US: FRED Census M3 aggregate + IP index sectors."""
    result: dict[str, Any] = {"country": "US", "ok": False}
    try:
        ipi      = _load_fred("IPMAN",         start=date(1990, 1, 1))
        ipi_yoy  = _compute_yoy(ipi).dropna()
        ship     = _load_fred("MNFCTRSMSA",    start=date(1992, 1, 1))
        inv      = _load_fred("MNFCTRIRSA",    start=date(1992, 1, 1))
        balance  = _compute_balance(ship, inv)
        activity = _load_fred("BSCICP02USM460S", start=date(2000, 1, 1))

        cfg_path = ROOT / "config" / "us" / "manufacturing.yaml"
        with open(cfg_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        sectors: dict[str, dict] = {}
        for sec in cfg.get("sectors", []):
            try:
                sid = sec["fred"]["series_id"]
                ip  = _load_fred(sid, start=date(2010, 1, 1))
                yoy = _compute_yoy(ip).dropna()
                st  = _sector_stats(yoy)
                if st is not None:
                    st["recovery_score"] = calc_recovery_score(yoy)
                    st["cycle_stage"]    = sec.get("cycle_stage", "mid")
                    sectors[sec["name_ja"]] = st
            except Exception as exc:
                log.warning("US sector %s skipped: %s", sec["name_ja"], exc)

        result.update({
            "ok": True,
            "ipi_yoy": ipi_yoy,
            "balance": balance,
            "phase":   detect_phase(balance),
            "sectors": sectors,
            "activity": activity,
            "activity_label": "US BCI (OECD)",
            "sector_note": "US sectors: IP production YoY proxy",
        })
    except Exception as exc:
        log.error("US load failed: %s", exc)
        result["error"] = str(exc)
    return result


def load_eu(start: date = date(2000, 1, 1)) -> dict[str, Any]:
    """EU: EA19 OECD IPI (ends 2023-10) + BCI."""
    result: dict[str, Any] = {"country": "EU", "ok": False}
    try:
        ipi_yoy  = _load_fred("EA19PRINTO01GYSAM", start=start)
        activity = _load_fred("BSCICP02EZM460S",   start=start)
        result.update({
            "ok": True,
            "ipi_yoy": ipi_yoy,
            "balance": pd.Series(dtype=float),
            "phase":   "データなし",
            "sectors": {},
            "activity": activity,
            "activity_label": "EU BCI (OECD)",
            "note": "EU: EA19 OECD series ends 2023-10",
        })
    except Exception as exc:
        log.error("EU load failed: %s", exc)
        result["error"] = str(exc)
    return result


def load_cn(start: date = date(2000, 1, 1)) -> dict[str, Any]:
    """CN: OECD BCI proxy only (no IPI/balance on FRED)."""
    result: dict[str, Any] = {"country": "CN", "ok": False}
    try:
        activity = _load_fred("CHNBSCICP02STSAM",  start=start)
        cli      = _load_fred("CHNLOLITONOSTSAM",   start=start)
        result.update({
            "ok": True,
            "ipi_yoy": pd.Series(dtype=float),
            "balance": pd.Series(dtype=float),
            "phase":   "データなし",
            "sectors": {},
            "activity": activity,
            "activity_label": "CN BCI (OECD)",
            "cli": cli,
            "note": "CN: No IPI/balance on FRED. BCI proxy only.",
        })
    except Exception as exc:
        log.error("CN load failed: %s", exc)
        result["error"] = str(exc)
    return result


_LOADERS = {"jp": load_jp, "us": load_us, "eu": load_eu, "cn": load_cn}


# ─────────────────────────────────────────────────────────────────────────────
# Summary JSON
# ─────────────────────────────────────────────────────────────────────────────

def _save_inv_cycle_summary(all_data: dict[str, dict], data_dir: Path) -> None:
    """
    在庫循環サマリーを data/inv_cycle_summary.json に保存する。
    prev_phase は「前回実行時の判定」を指す（前月ではなく前回 run() 時点）。
    """
    out_path = data_dir / "inv_cycle_summary.json"

    existing: dict[str, dict] = {}
    if out_path.exists():
        try:
            with open(out_path, encoding="utf-8") as f:
                existing = json.load(f)
        except Exception:
            pass

    summary: dict[str, Any] = {"generated_at": date.today().isoformat()}

    for c, d in all_data.items():
        if not d.get("ok"):
            summary[c] = {"ok": False, "error": d.get("error", "unknown")}
            continue

        phase        = d.get("phase", "データなし")
        prev_phase   = existing.get(c, {}).get("phase")
        phase_changed = (prev_phase is not None) and (prev_phase != phase)

        df   = d.get("df_sectors")
        top3: list[dict] = []
        bot3: list[dict] = []
        if df is not None and not df.empty and "recovery_score" in df.columns:
            ranked = df["recovery_score"].dropna().sort_values()
            top3 = [
                {"name": nm, "score": round(float(score), 1)}
                for nm, score in ranked.tail(3).iloc[::-1].items()
            ]
            bot3 = [
                {"name": nm, "score": round(float(score), 1)}
                for nm, score in ranked.head(3).items()
            ]

        summary[c] = {
            "ok":            True,
            "phase":         phase,
            "prev_phase":    prev_phase,
            "phase_changed": phase_changed,
            "top3_recovery": top3,
            "bot3_recovery": bot3,
        }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    log.info("在庫循環サマリー保存: %s", out_path)


# ─────────────────────────────────────────────────────────────────────────────
# Charts
# ─────────────────────────────────────────────────────────────────────────────

def plot_activity_comparison(
    all_data: dict[str, dict],
    output_path: Path,
    start_year: int = 2018,
) -> None:
    """Global manufacturing activity comparison (BCI/CLI proxies)."""
    _set_jp_font()
    fig, ax = plt.subplots(figsize=(10, 5))

    start_dt = pd.Timestamp(f"{start_year}-01-01")

    for country, d in all_data.items():
        if not d.get("ok") or d.get("activity") is None:
            continue
        act = d["activity"].dropna()
        if act.empty:
            continue
        # z-score normalize so CLI (~100) and BCI (~0) are comparable
        mu, sigma = float(act.mean()), float(act.std())
        if sigma > 0:
            act = (act - mu) / sigma
        act = act.loc[act.index >= start_dt]
        if act.empty:
            continue
        color = _COUNTRY_COLORS.get(country.upper(), "#888888")
        label = f"{country.upper()} — {d.get('activity_label', '')}"
        ax.plot(act.index, act.values, color=color, linewidth=1.8, label=label)

    ax.axhline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.5)
    ax.set_title("製造業活動指標 (BCI/CLI 代替)", fontsize=12, fontweight="bold")
    ax.set_ylabel("標準化 (z-score)", fontsize=9)
    ax.legend(fontsize=8, loc="upper left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.xticks(rotation=30, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.text(
        0.5, 0.01,
        "Source: OECD/FRED — BCI (US/EU/CN) and CLI (JP) proxy for manufacturing PMI  |  各系列z-score標準化",
        ha="center", fontsize=7, color="gray",
    )
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("出力: %s", output_path)


def plot_timeseries(
    data: dict,
    output_path: Path,
    start_year: int = 2018,
) -> None:
    """IPI YoY + 出荷在庫バランス time series for one country."""
    _set_jp_font()
    country = data.get("country", "??")
    ipi_yoy = data.get("ipi_yoy", pd.Series(dtype=float))
    balance = data.get("balance", pd.Series(dtype=float))

    start_dt = pd.Timestamp(f"{start_year}-01-01")

    has_ipi = (ipi_yoy is not None) and (not ipi_yoy.empty)
    has_bal = (balance is not None) and (not balance.empty)
    n_panels = 2 if (has_ipi and has_bal) else 1

    fig, axes = plt.subplots(n_panels, 1, figsize=(10, 3.5 * n_panels), sharex=True)
    if n_panels == 1:
        axes = [axes]

    # Fallback: when neither IPI nor balance available, plot activity series
    activity     = data.get("activity")
    has_activity = (activity is not None) and (not activity.empty)
    use_fallback = (not has_ipi) and (not has_bal) and has_activity

    if use_fallback:
        ax = axes[0]
        s = activity.loc[activity.index >= start_dt].dropna()
        # z-score so scale is interpretable alongside other charts
        mu, sigma = float(s.mean()), float(s.std())
        if sigma > 0:
            s = (s - mu) / sigma
        color = _COUNTRY_COLORS.get(country, "#555555")
        ax.plot(s.index, s.values, color=color, linewidth=1.8)
        ax.axhline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.5)
        label = data.get("activity_label", "Activity")
        note  = data.get("note", "")
        title_note = f"  [{note}]" if note else ""
        ax.set_title(f"{country} — {label} (z-score標準化){title_note}", fontsize=10)
        ax.set_ylabel("標準化 (z-score)", fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    panel = 0

    # IPI YoY panel
    if has_ipi and not use_fallback:
        ax = axes[panel]
        s = ipi_yoy.loc[ipi_yoy.index >= start_dt].dropna()
        ax.plot(s.index, s.values, color=_COUNTRY_COLORS.get(country, "#555555"),
                linewidth=1.8)
        ax.axhline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.5)
        ax.set_ylabel("IPI YoY (%)", fontsize=9)
        note = data.get("note", "")
        title_note = f"  [{note}]" if note else ""
        ax.set_title(f"{country} — 鉱工業生産指数 前年同月比{title_note}", fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        panel += 1

    # Balance panel
    if has_bal and not use_fallback:
        ax = axes[panel]
        s = balance.loc[balance.index >= start_dt].dropna()
        colors = ["#27ae60" if v >= 0 else "#e74c3c" for v in s.values]
        ax.bar(s.index, s.values, color=colors, width=25, alpha=0.85)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_ylabel("出荷在庫バランス (%pt)", fontsize=9)
        ax.set_title(f"{country} — 出荷在庫バランス (出荷YoY - 在庫YoY)", fontsize=10)
        ax.grid(axis="y", alpha=0.3)

        # Current phase annotation
        phase = data.get("phase", "")
        if phase and phase not in ("不明", "データなし"):
            color = _PHASE_COLORS.get(phase, "#888888")
            ax.text(
                0.98, 0.95, f"現状: {phase}",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=9, color="white",
                bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.85),
            )
        panel += 1

    # Shared x-axis formatting
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axes[-1].xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.xticks(rotation=30, ha="right", fontsize=8)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("出力: %s", output_path)


def plot_sector_snapshot(
    data: dict,
    output_path: Path,
) -> None:
    """Sector snapshot: floating range bars + red polylines (Mizuho-style)."""
    _set_jp_font()
    sectors = data.get("sectors", {})
    if not sectors:
        log.info("No sector data for %s, skipping", data.get("country"))
        return

    country = data.get("country", "??")

    # Sort sectors: early → mid → late (stable: preserves YAML order within stage)
    _STAGE_ORDER = {"early": 0, "mid": 1, "late": 2}
    names = sorted(
        sectors.keys(),
        key=lambda nm: _STAGE_ORDER.get(sectors[nm].get("cycle_stage", "mid"), 1),
    )
    n = len(names)
    x = np.arange(n)

    peaks    = np.array([sectors[nm].get("peak",    float("nan")) for nm in names])
    troughs  = np.array([sectors[nm].get("trough",  float("nan")) for nm in names])
    q4_vals  = np.array([sectors[nm].get("q4",      float("nan")) for nm in names])
    cur_vals = np.array([sectors[nm].get("current", float("nan")) for nm in names])

    fig, ax = plt.subplots(figsize=(max(12, n * 1.1), 6))

    # ── 0. cycle_stage background shading (no text labels — keep shading only) ──
    _STAGE_BG = {"early": "#3b82f6", "mid": "#f97316", "late": "#22c55e"}

    prev_stage = None
    seg_start  = 0
    segments: list[tuple[int, int, str]] = []
    for i, nm in enumerate(names):
        stage = sectors[nm].get("cycle_stage", "mid")
        if stage != prev_stage:
            if prev_stage is not None:
                segments.append((seg_start, i, prev_stage))
            seg_start  = i
            prev_stage = stage
    if prev_stage is not None:
        segments.append((seg_start, n, prev_stage))

    for seg_s, seg_e, stage in segments:
        ax.axvspan(seg_s - 0.5, seg_e - 0.5,
                   color=_STAGE_BG.get(stage, "#888888"), alpha=0.07, zorder=0)

    # ── 1. Floating range bars (trough → peak, water-blue) ───────────────
    for i in range(n):
        p, b = peaks[i], troughs[i]
        if not (np.isnan(p) or np.isnan(b)):
            ax.bar(x[i], height=p - b, bottom=b,
                   color="#93c5fd", alpha=0.45, width=0.8, zorder=1,
                   linewidth=0)

    # ── 2. Zero line ──────────────────────────────────────────────────────
    ax.axhline(0, color="black", linewidth=0.8, zorder=2)

    # ── 3. Peak / trough open-circle markers (on bar edges) ──────────────
    ax.scatter(x, peaks,   s=70, facecolor="white", edgecolor="#1d4ed8",
               linewidths=2.0, zorder=4, label="直近ピーク")
    ax.scatter(x, troughs, s=70, facecolor="white", edgecolor="#dc2626",
               linewidths=2.0, zorder=4, label="直近ボトム")

    # ── 4. Red polylines: Q-4 (small ●) and current (large ●) ───────────
    ax.plot(x, q4_vals,  color="#dc2626", linewidth=1.3,
            marker="o", markersize=6,  zorder=5, label="4四半期前")
    ax.plot(x, cur_vals, color="#dc2626", linewidth=1.8,
            marker="o", markersize=10, zorder=6, label="現在")

    # ── 5. Recovery score annotations on current points ──────────────────
    for i, nm in enumerate(names):
        rs = sectors[nm].get("recovery_score", float("nan"))
        cv = cur_vals[i]
        if np.isnan(rs) or np.isnan(cv):
            continue
        y_dir = 1 if cv >= 0 else -1
        ax.annotate(
            f"{rs:.0f}pt",
            xy=(x[i], cv),
            xytext=(0, y_dir * 8),
            textcoords="offset points",
            ha="center",
            va="bottom" if y_dir > 0 else "top",
            fontsize=7,
            color="#14532d",
            fontweight="bold",
        )

    # ── 6. Axes, labels, legend ───────────────────────────────────────────
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=90, ha="center", fontsize=8)
    ax.set_ylabel("(%pt)", fontsize=9)
    ax.yaxis.set_label_coords(-0.04, 1.02)
    ax.yaxis.label.set_rotation(0)
    sector_note = data.get("sector_note", "出荷在庫バランス")
    ax.set_title(
        f"{country} — セクター別在庫循環スナップショット ({sector_note})",
        fontsize=10, fontweight="bold",
    )
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("出力: %s", output_path)


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def run(
    countries: list[str] | None = None,
    start_year: int = 2018,
) -> dict[str, dict]:
    """
    Load data and produce all charts for requested countries.

    Returns dict[country_code → result_dict].
    """
    if countries is None:
        countries = ["jp", "us", "eu", "cn"]

    data_dir = ROOT / "data"

    all_data: dict[str, dict] = {}
    for c in countries:
        loader = _LOADERS.get(c.lower())
        if loader is None:
            log.warning("No loader for country %s, skipping", c)
            continue
        log.info("Loading %s ...", c.upper())
        d = loader()
        all_data[c.upper()] = d
        if not d["ok"]:
            log.warning("%s load failed: %s", c.upper(), d.get("error"))

    # Build df_sectors per country (includes recovery_score column)
    for d in all_data.values():
        sectors = d.get("sectors", {})
        if sectors:
            df = pd.DataFrame.from_dict(sectors, orient="index")
            df.index.name = "sector"
        else:
            df = pd.DataFrame(columns=["peak", "trough", "q4", "current", "recovery_score"])
            df.index.name = "sector"
        d["df_sectors"] = df

    # Global activity comparison
    act_path = data_dir / "inv_cycle_activity.png"
    try:
        plot_activity_comparison(all_data, act_path, start_year=start_year)
    except Exception as exc:
        log.error("Activity comparison chart failed: %s", exc)

    # Per-country charts
    for c, d in all_data.items():
        if not d["ok"]:
            continue
        country_dir = data_dir / c.lower()
        country_dir.mkdir(parents=True, exist_ok=True)

        try:
            plot_timeseries(d, country_dir / "inv_cycle_ts.png", start_year=start_year)
        except Exception as exc:
            log.error("%s timeseries chart failed: %s", c, exc)

        try:
            plot_sector_snapshot(d, country_dir / "inv_cycle_sector.png")
        except Exception as exc:
            log.error("%s sector chart failed: %s", c, exc)

    try:
        _save_inv_cycle_summary(all_data, data_dir)
    except Exception as exc:
        log.error("inv_cycle_summary.json 保存失敗: %s", exc)

    return all_data
