"""
エンドツーエンド実行スクリプト(MVP 4指標版)。

実行:
    python scripts/01_jp_mvp.py

出力:
    - data/level_momentum.parquet : 月次のレベル/モメンタム/ステージ
    - data/stage_timeline.png     : 1985年以降のステージ推移(ステージ色帯付き)
"""
from __future__ import annotations

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
CFG_INDICATORS = ROOT / "config" / "indicators.yaml"
CFG_SETTINGS   = ROOT / "config" / "settings.yaml"
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

STAGE_COLORS = {
    "回復": "#fffacd",   # lemon chiffon
    "上昇": "#90ee90",   # light green
    "成熟": "#add8e6",   # light blue
    "軟化": "#ffd700",   # gold
    "下降": "#ffb6c1",   # light pink / red
}
# English labels for chart legend (CJK font not guaranteed)
STAGE_EN = {
    "回復": "Recovery",
    "上昇": "Expansion",
    "成熟": "Mature",
    "軟化": "Softening",
    "下降": "Decline",
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
    """ステージごとに背景色帯を描く。"""
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


def main() -> None:
    with open(CFG_SETTINGS, "r", encoding="utf-8") as f:
        settings = yaml.safe_load(f)

    # ── 指標取得 ──────────────────────────────────────────────
    indicators = get_all_indicators(CFG_INDICATORS, start=date(1985, 1, 1))

    # ── 派生指標を計算 ────────────────────────────────────────
    derived = compute_derived_indicators(indicators, CFG_INDICATORS)
    indicators.update(derived)

    print(f"\n取得指標: {sorted(indicators.keys())} ({len(indicators)}本)")

    # ── 各指標の観測期間と直近値をレポート ────────────────────
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

    # ── 符号反転 ──────────────────────────────────────────────
    for inv_id in settings["composite"].get("invert_sign", []):
        if inv_id in indicators:
            indicators[inv_id] = -indicators[inv_id]

    # ── 合成 ─────────────────────────────────────────────────
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

    # ── ステージ判定 ──────────────────────────────────────────
    sc = settings["stage_classification"]
    stage = classify_stage(
        level, momentum,
        level_low=sc["level_low"],
        level_high=sc["level_high"],
        momentum_threshold=sc["momentum_threshold"],
        hysteresis_periods=sc["hysteresis_periods"],
    )

    out = pd.DataFrame({"level": level, "momentum": momentum, "stage": stage})

    # ── テーブル出力 ──────────────────────────────────────────
    recent_12 = out.dropna(subset=["level"]).tail(12)
    _print_table(recent_12, "直近12ヶ月 (Level / Momentum / Stage)")

    gfc = out.loc["2008-09":"2009-03"].dropna(subset=["level"])
    _print_table(gfc, "GFC リセッション: 2008-09 ～ 2009-03")

    covid = out.loc["2020-02":"2020-06"].dropna(subset=["level"])
    _print_table(covid, "コロナ: 2020-02 ～ 2020-06")

    # ── ステージ分布サマリ ─────────────────────────────────────
    valid = out["stage"].dropna()
    if len(valid):
        print(f"\n{'='*64}")
        print("  ステージ分布 (全期間)")
        print("="*64)
        cnt = valid.value_counts().reindex(["回復","上昇","成熟","軟化","下降"]).fillna(0)
        for stg, n in cnt.items():
            print(f"  {stg}: {int(n):4d} ヶ月 ({100*n/len(valid):.1f}%)")

    # ── 保存 ─────────────────────────────────────────────────
    out.to_parquet(DATA_DIR / "level_momentum.parquet")

    # ── チャート ──────────────────────────────────────────────
    valid_out = out.dropna(subset=["level"])
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    fig.suptitle("Japan Business Cycle MVP (7-indicator)", fontsize=13, fontweight="bold")

    ax0, ax1 = axes

    _shade_stages(ax0, valid_out["stage"])
    ax0.plot(valid_out.index, valid_out["level"], color="#333333", linewidth=1.5, zorder=3)
    ax0.axhline(sc["level_low"],  color="#2ca02c", linestyle="--", alpha=0.7, linewidth=1)
    ax0.axhline(sc["level_high"], color="#d62728", linestyle="--", alpha=0.7, linewidth=1)
    ax0.set_ylim(0, 100)
    ax0.set_ylabel("Cycle Level (0–100)")
    ax0.set_yticks([0, 25, 50, 75, 100])

    _shade_stages(ax1, valid_out["stage"])
    ax1.plot(valid_out.index, valid_out["momentum"], color="steelblue", linewidth=1.5, zorder=3)
    ax1.axhline(0, color="black", linestyle="--", alpha=0.4, linewidth=1)
    ax1.set_ylabel("Momentum (z-score)")

    # 凡例 (英語のみ)
    patches = [mpatches.Patch(color=c, alpha=0.55, label=STAGE_EN[s])
               for s, c in STAGE_COLORS.items()]
    ax0.legend(handles=patches, loc="upper left", fontsize=8, ncol=5,
               framealpha=0.8)

    plt.tight_layout()
    png_path = DATA_DIR / "stage_timeline.png"
    plt.savefig(png_path, dpi=150)
    print(f"\n出力: {png_path}")
    print(f"出力: {DATA_DIR / 'level_momentum.parquet'}")


if __name__ == "__main__":
    main()
