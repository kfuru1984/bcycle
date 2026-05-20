"""
景気循環モデル 汎用実行スクリプト。

実行:
    python scripts/run_cycle.py --country jp
    python scripts/run_cycle.py --country us

出力 (data/{country}/ 以下):
    - level_momentum.parquet
    - stage_timeline.png
    - cycle_detail.json   ← 新規: ダッシュボード用詳細データ
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
import yaml

from bcycle_jp.core.classify import classify_stage
from bcycle_jp.core.composite import compute_level, compute_momentum
from bcycle_jp.core.loader import compute_derived_indicators, get_all_indicators
from bcycle_jp.core.normalize import rolling_percentile, rolling_zscore

ROOT = Path(__file__).resolve().parents[1]

STAGE_COLORS = {
    "回復": "#fffacd",
    "上昇": "#90ee90",
    "成熟": "#add8e6",
    "軟化": "#ffd700",
    "下降": "#ffb6c1",
}
STAGE_EN = {
    "回復": "Recovery",
    "上昇": "Expansion",
    "成熟": "Mature",
    "軟化": "Softening",
    "下降": "Decline",
}

RECESSION_CHECKS: dict[str, list[tuple[str, str, str]]] = {
    "jp": [
        ("GFC", "2008-09", "2009-03"),
        ("COVID", "2020-02", "2020-06"),
    ],
    "us": [
        ("GFC", "2008-09", "2009-06"),
        ("COVID", "2020-02", "2020-04"),
    ],
    "kr": [
        ("AFC", "1997-11", "1998-12"),
        ("GFC", "2008-09", "2009-02"),
        ("COVID", "2020-02", "2020-04"),
    ],
}

START_DATES: dict[str, date] = {
    "jp": date(1985, 1, 1),
    "us": date(1960, 1, 1),
    "kr": date(1990, 1, 1),
}

CHART_TITLES: dict[str, str] = {
    "jp": "Japan Business Cycle (7-indicator)",
    "us": "US Business Cycle (9-indicator)",
    "kr": "Korea Business Cycle",
}

CATEGORY_JA: dict[str, str] = {
    "production": "生産",
    "labor": "労働",
    "housing": "住宅",
    "consumption": "消費",
    "external": "外需",
    "inflation": "物価",
    "financial": "金融",
    "investment": "設備投資",
    "sentiment": "信頼感",
    "other": "その他",
}


def _print_table(df: pd.DataFrame, title: str) -> None:
    print(f"\n{'='*64}")
    print(f"  {title}")
    print("="*64)
    tbl = df[["level", "momentum", "stage"]].copy()
    tbl.index = tbl.index.strftime("%Y-%m")
    tbl["level"]    = tbl["level"].map(lambda x: f"{x:6.1f}" if pd.notna(x) else "   NaN")
    tbl["momentum"] = tbl["momentum"].map(lambda x: f"{x:+.3f}" if pd.notna(x) else "   NaN")
    tbl["stage"]    = tbl["stage"].fillna("-")
    print(tbl.to_string())


def _shade_stages(ax, stage: pd.Series) -> None:
    if stage.empty:
        return
    dates = stage.index
    prev_stg = stage.iloc[0]
    seg_start = dates[0]
    for i in range(1, len(stage)):
        if stage.iloc[i] != prev_stg:
            color = STAGE_COLORS.get(prev_stg, "#f0f0f0")
            ax.axvspan(seg_start, dates[i], alpha=0.35, color=color, linewidth=0)
            seg_start = dates[i]
            prev_stg = stage.iloc[i]
    color = STAGE_COLORS.get(prev_stg, "#f0f0f0")
    ax.axvspan(seg_start, dates[-1], alpha=0.35, color=color, linewidth=0)


def _get_last_val(s: pd.Series, as_of) -> float | None:
    sub = s.loc[:as_of].dropna()
    return float(sub.iloc[-1]) if not sub.empty else None


def build_cycle_detail(
    country: str,
    indicators_orig: dict[str, pd.Series],
    indicators_inv: dict[str, pd.Series],
    weights: dict[str, float],
    cfg_indicators: Path,
    norm: dict,
    level_series: pd.Series,
    momentum_series: pd.Series,
    stage_series: pd.Series,
) -> dict:
    """Compute per-indicator stats and return dict for cycle_detail.json."""
    window   = norm["rolling_window_months"]
    min_p    = norm["min_periods_months"]

    with open(cfg_indicators, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    ind_meta = {ind["id"]: ind for ind in cfg["indicators"]}

    valid = pd.DataFrame({"level": level_series, "momentum": momentum_series, "stage": stage_series})
    valid = valid.dropna(subset=["level"])
    if valid.empty:
        return {"ok": False, "country": country.upper()}

    last_row = valid.iloc[-1]
    as_of    = valid.index[-1]
    as_of_str = str(as_of)[:7]

    # ── Per-indicator z / percentile (on inverted series) ─────────────
    active_ids = [k for k, v in weights.items() if v > 0 and k in indicators_inv]
    per_ind: dict[str, dict] = {}

    for ind_id in active_ids:
        s_inv = indicators_inv[ind_id].dropna()
        if s_inv.empty:
            continue
        pct_s = rolling_percentile(s_inv, window, min_p)
        z_s   = rolling_zscore(s_inv, window, min_p)

        pct_val = _get_last_val(pct_s, as_of)
        z_val   = _get_last_val(z_s,   as_of)
        # Display value: pre-invert (original)
        orig_val = _get_last_val(indicators_orig.get(ind_id, s_inv), as_of)

        meta = ind_meta.get(ind_id, {})
        per_ind[ind_id] = {
            "id":        ind_id,
            "name_ja":   meta.get("name_ja") or meta.get("name_en", ind_id),
            "category":  meta.get("category", "other"),
            "percentile": round(pct_val * 100, 1) if pct_val is not None else None,
            "z":          round(z_val,  2)         if z_val   is not None else None,
            "value":      round(orig_val, 3)        if orig_val is not None else None,
            "transform":  meta.get("transform", "level"),
        }

    valid_for_sort = [v for v in per_ind.values() if v["percentile"] is not None]
    valid_for_z    = [v for v in per_ind.values() if v["z"]          is not None]
    top3_level    = sorted(valid_for_sort, key=lambda x: x["percentile"], reverse=True)[:3]
    top3_momentum = sorted(valid_for_z,    key=lambda x: abs(x["z"]),     reverse=True)[:3]

    # ── CB stance ─────────────────────────────────────────────────────
    rpr = _get_last_val(indicators_orig.get("real_policy_rate", pd.Series(dtype=float)), as_of)
    yc  = _get_last_val(indicators_orig.get("yc_10y2y",         pd.Series(dtype=float)), as_of)

    if rpr is None:
        cb_stance = "データなし"
        cb_detail = ""
    elif rpr < -1.0:
        cb_stance = "ビハインド・ザ・カーブ"
        cb_detail = f"Taylor比 {rpr:.1f}%下、利上げ遅れ"
    elif rpr < 0.0:
        cb_stance = "ややビハインド"
        cb_detail = f"Taylor比 {rpr:.1f}%下、緩和的"
    elif rpr < 1.0:
        cb_stance = "中立"
        cb_detail = f"実質金利 {rpr:+.1f}%"
    else:
        cb_stance = "引き締め気味"
        cb_detail = f"実質金利 {rpr:+.1f}%"

    if yc is not None:
        yc_label = f"YC {yc:+.2f}%"
        cb_detail = (cb_detail + " / " + yc_label) if cb_detail else yc_label

    # ── Real wage ─────────────────────────────────────────────────────
    real_wage = None
    if "nominal_wage_yoy" in indicators_orig and "core_cpi_yoy" in indicators_orig:
        nw_s  = indicators_orig["nominal_wage_yoy"].dropna()
        cpi_s = indicators_orig["core_cpi_yoy"].dropna()
        rw_s  = (nw_s - cpi_s).dropna()
        if not rw_s.empty:
            rw_last = _get_last_val(rw_s.to_frame(), as_of) if False else (
                float(rw_s.loc[:as_of].iloc[-1]) if not rw_s.loc[:as_of].empty else None
            )
            real_wage = round(rw_last, 2) if rw_last is not None else None

    # ── Core CPI for outlook display ──────────────────────────────────
    core_cpi_val = _get_last_val(indicators_orig.get("core_cpi_yoy", pd.Series(dtype=float)), as_of)

    return {
        "ok":              True,
        "country":         country.upper(),
        "as_of":           as_of_str,
        "stage":           str(last_row["stage"]),
        "level":           round(float(last_row["level"]),    1),
        "momentum":        round(float(last_row["momentum"]), 2),
        "cb_stance":       cb_stance,
        "cb_detail":       cb_detail,
        "real_policy_rate": round(rpr, 2) if rpr is not None else None,
        "yc_10y2y":        round(yc,  2) if yc  is not None else None,
        "real_wage_yoy":   real_wage,
        "core_cpi_yoy":    round(core_cpi_val, 2) if core_cpi_val is not None else None,
        "top3_level":      top3_level,
        "top3_momentum":   top3_momentum,
        "per_indicators":  per_ind,
    }


def fetch_cn_signals(cfg_path: Path) -> list[dict]:
    from datetime import date as _date
    from bcycle_jp.adapters.fred import FredAdapter

    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    fred = FredAdapter()
    results: list[dict] = []
    for sig in cfg.get("signals", []):
        try:
            series = fred.fetch({"series_id": sig["series_id"]}, start=_date(2000, 1, 1))
            series = series.dropna()
            last_val  = float(series.iloc[-1])
            last_date = series.index[-1]
            threshold = float(sig.get("threshold", 0.0))
            label = sig["expanding_label"] if last_val >= threshold else sig["contracting_label"]
            results.append({
                "id":    sig["id"],
                "name":  sig["name_en"],
                "value": last_val,
                "date":  last_date,
                "label": label,
                "note":  sig.get("name_ja", ""),
            })
        except Exception as exc:
            results.append({"id": sig["id"], "name": sig.get("name_en", "?"), "error": str(exc)})
    return results


def print_signals_summary(countries: list[str], cn_results: list[dict], data_root: Path) -> None:
    print(f"\n{'='*64}")
    print("  Cycle Summary")
    print("="*64)

    for c in countries:
        parquet_path = data_root / c / "level_momentum.parquet"
        if not parquet_path.exists():
            print(f"  {c.upper()}: (データなし)")
            continue
        df = pd.read_parquet(parquet_path)
        valid = df.dropna(subset=["level"])
        if valid.empty:
            print(f"  {c.upper()}: (有効データなし)")
            continue
        last     = valid.iloc[-1]
        stage    = last["stage"] if pd.notna(last.get("stage")) else "-"
        level    = last["level"]
        momentum = last["momentum"]
        date_str = str(valid.index[-1])[:7]
        print(f"  {c.upper()}: {stage:<4}  (level={level:.1f})  Momentum: {momentum:+.2f}  [{date_str}]")

    print("-"*64)

    for r in cn_results:
        if "error" in r:
            print(f"  CN Signal [{r['id']}]: 取得失敗 ({r['error']})")
        else:
            date_str = str(r["date"])[:7]
            print(f"  CN Signal: BCI {r['value']:.2f} ({r['label']})  [{date_str}]")

    print("="*64)


def main(country: str) -> None:
    cfg_indicators = ROOT / "config" / country / "indicators.yaml"
    cfg_settings   = ROOT / "config" / country / "settings.yaml"
    data_dir       = ROOT / "data" / country
    data_dir.mkdir(parents=True, exist_ok=True)

    with open(cfg_settings, "r", encoding="utf-8") as f:
        settings = yaml.safe_load(f)

    start = START_DATES.get(country, date(1980, 1, 1))
    indicators = get_all_indicators(cfg_indicators, start=start)

    derived = compute_derived_indicators(indicators, cfg_indicators)
    indicators.update(derived)

    print(f"\n取得指標: {sorted(indicators.keys())} ({len(indicators)}本)")

    print(f"\n{'='*64}")
    print("  指標別 観測期間・直近値")
    print("="*64)
    for name in sorted(indicators.keys()):
        s = indicators[name].dropna()
        if s.empty:
            print(f"  {name:<30}: (データなし)")
        else:
            print(
                f"  {name:<30}: "
                f"{str(s.index[0])[:7]} - {str(s.index[-1])[:7]}  "
                f"({len(s)} obs)  直近={s.iloc[-1]:.3f}"
            )

    # Save pre-invert copy for display values in cycle_detail
    indicators_orig = {k: v.copy() for k, v in indicators.items()}

    for inv_id in settings["composite"].get("invert_sign", []):
        if inv_id in indicators:
            indicators[inv_id] = -indicators[inv_id]

    norm = settings["normalize"]
    level = compute_level(
        indicators,
        weights=settings["composite"]["weights"],
        method=settings["composite"]["method"],
        window=norm["rolling_window_months"],
        min_periods=norm["min_periods_months"],
        min_indicator_frac=settings["composite"].get("min_indicator_frac", 0.5),
    )

    momentum = compute_momentum(
        level,
        lookback=settings["momentum"]["lookback_months"],
        window=norm["rolling_window_months"],
        min_periods=norm["min_periods_months"],
    )

    sc = settings["stage_classification"]
    stage = classify_stage(
        level, momentum,
        level_low=sc["level_low"],
        level_high=sc["level_high"],
        momentum_threshold=sc["momentum_threshold"],
        hysteresis_periods=sc["hysteresis_periods"],
    )

    out = pd.DataFrame({"level": level, "momentum": momentum, "stage": stage})

    recent_12 = out.dropna(subset=["level"]).tail(12)
    _print_table(recent_12, "直近12ヶ月 (Level / Momentum / Stage)")

    for label, start_str, end_str in RECESSION_CHECKS.get(country, []):
        period = out.loc[start_str:end_str].dropna(subset=["level"])
        _print_table(period, f"{label}: {start_str} ~ {end_str}")

    valid = out["stage"].dropna()
    if len(valid):
        print(f"\n{'='*64}")
        print("  ステージ分布 (全期間)")
        print("="*64)
        cnt = valid.value_counts().reindex(["回復", "上昇", "成熟", "軟化", "下降"]).fillna(0)
        for stg, n in cnt.items():
            print(f"  {stg}: {int(n):4d} ヶ月 ({100*n/len(valid):.1f}%)")

    parquet_path = data_dir / "level_momentum.parquet"
    out.to_parquet(parquet_path)

    # ── cycle_detail.json ──────────────────────────────────────────────
    detail = build_cycle_detail(
        country=country,
        indicators_orig=indicators_orig,
        indicators_inv=indicators,
        weights=settings["composite"]["weights"],
        cfg_indicators=cfg_indicators,
        norm=norm,
        level_series=level,
        momentum_series=momentum,
        stage_series=stage,
    )
    detail_path = data_dir / "cycle_detail.json"
    detail_path.write_text(
        json.dumps(detail, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"\n出力: {detail_path}")

    # ── Stage timeline chart ───────────────────────────────────────────
    valid_out = out.dropna(subset=["level"])
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    fig.suptitle(CHART_TITLES.get(country, f"{country.upper()} Business Cycle"),
                 fontsize=13, fontweight="bold")

    ax0, ax1 = axes

    _shade_stages(ax0, valid_out["stage"])
    ax0.plot(valid_out.index, valid_out["level"], color="#333333", linewidth=1.5, zorder=3)
    ax0.axhline(sc["level_low"],  color="#2ca02c", linestyle="--", alpha=0.7, linewidth=1)
    ax0.axhline(sc["level_high"], color="#d62728", linestyle="--", alpha=0.7, linewidth=1)
    ax0.set_ylim(0, 100)
    ax0.set_ylabel("Cycle Level (0-100)")
    ax0.set_yticks([0, 25, 50, 75, 100])

    _shade_stages(ax1, valid_out["stage"])
    ax1.plot(valid_out.index, valid_out["momentum"], color="steelblue", linewidth=1.5, zorder=3)
    ax1.axhline(0, color="black", linestyle="--", alpha=0.4, linewidth=1)
    ax1.set_ylabel("Momentum (z-score)")

    patches = [mpatches.Patch(color=c, alpha=0.55, label=STAGE_EN[s])
               for s, c in STAGE_COLORS.items()]
    ax0.legend(handles=patches, loc="upper left", fontsize=8, ncol=5, framealpha=0.8)

    plt.tight_layout()
    png_path = data_dir / "stage_timeline.png"
    plt.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"出力: {png_path}")
    print(f"出力: {parquet_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="景気循環モデル実行")
    parser.add_argument("--country", choices=["jp", "us", "kr"],
                        help="国コード: jp / us / kr")
    parser.add_argument("--signals", action="store_true",
                        help="CN外部シグナルと全カ国サマリーを表示")
    args = parser.parse_args()

    if not args.country and not args.signals:
        parser.error("--country または --signals が必要")

    if args.country:
        main(args.country)

    if args.signals:
        cfg_cn = ROOT / "config" / "cn_signals.yaml"
        cn_results = fetch_cn_signals(cfg_cn)
        countries  = ["jp", "us", "kr"]
        print_signals_summary(countries, cn_results, ROOT / "data")
