"""
景気循環モデル 汎用実行スクリプト。

実行:
    python scripts/run_cycle.py --country jp
    python scripts/run_cycle.py --country us

出力 (data/{country}/ 以下):
    - level_momentum.parquet
    - stage_timeline.png
"""
from __future__ import annotations

import argparse
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


def fetch_cn_signals(cfg_path: Path) -> list[dict]:
    """cn_signals.yaml を読み、各シグナルの直近値を取得して返す。"""
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
    """全カ国のサイクルサマリーと CN シグナルをまとめて表示。"""
    print(f"\n{'='*64}")
    print("  Cycle Summary")
    print("="*64)

    for c in countries:
        parquet_path = data_root / c / "level_momentum.parquet"
        if not parquet_path.exists():
            print(f"  {c.upper()}: (データなし — run_cycle.py --country {c} を先に実行)")
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
            print(f"  CN Signal [{r['id']}]: 取得失敗 — {r['error']}")
        else:
            date_str = str(r["date"])[:7]
            print(f"  CN Signal: BCI {r['value']:.2f} ({r['label']})  [{date_str}]")
            print(f"             ※ {r['note']}")
            print(f"             ※ NBS/Caixin PMI は FRED/OECD で無償取得不可")

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
    print(f"\n出力: {png_path}")
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
