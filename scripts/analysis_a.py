"""
Analysis A: ステージ別セクター/ファクターリターン。

JP: Japan 5-Factors + Japan Momentum (6 factors, already excess returns)
    → data/jp/stage_factor_heatmap.png

US: 12 Industry Portfolios VW monthly + RF from FF 5-Factors
    → data/us/stage_sector_heatmap.png

KR: Ken French にKorea個別データなし → スキップ

実行:
    python scripts/analysis_a.py            # JP + US
    python scripts/analysis_a.py --country jp
    python scripts/analysis_a.py --country us
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from bcycle_jp.adapters import french
from bcycle_jp.analysis.stage_returns import compute_stage_stats, plot_stage_heatmap

ROOT = Path(__file__).resolve().parents[1]


def _load_stage(country: str) -> pd.Series:
    path = ROOT / "data" / country / "level_momentum.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} が見つかりません。先に run_cycle.py --country {country} を実行してください。"
        )
    df = pd.read_parquet(path)
    return df["stage"].dropna()


def _sample_label(series_index: pd.DatetimeIndex) -> str:
    return f"{str(series_index[0])[:7]} ~ {str(series_index[-1])[:7]}"


# ─────────────────────────────────────────────────────────────────────────────
# JP: Factor analysis
# ─────────────────────────────────────────────────────────────────────────────

def run_jp() -> None:
    print("\n=== JP: ステージ別ファクターリターン ===")

    stage = _load_stage("jp")
    print(f"  Stage series: {_sample_label(stage.index)}  ({len(stage)} obs)")

    # Japan 5 Factors (Mkt-RF, SMB, HML, RMW, CMA, RF)
    ff5 = french.fetch("Japan_5_Factors_CSV.zip")
    print(f"  JP 5-Factors: {_sample_label(ff5.index)}  ({len(ff5)} obs)")

    # Japan Momentum (WML)
    mom = french.fetch("Japan_Mom_Factor_CSV.zip")
    mom = mom.rename(columns={"WML": "MOM"})
    print(f"  JP Momentum:  {_sample_label(mom.index)}  ({len(mom)} obs)")

    # Combine factors; drop RF (not needed — all factors are zero-cost or already excess)
    factors = ff5.drop(columns=["RF"], errors="ignore").join(mom[["MOM"]], how="inner")
    print(f"  Combined:     {_sample_label(factors.index)}  ({len(factors)} obs)  cols={list(factors.columns)}")

    mean_ret, tstat, nobs = compute_stage_stats(stage, factors, rf_series=None)

    # Print summary
    print("\n  Mean monthly excess return (%) by stage:")
    print(f"  {'Stage':<6}", end="")
    for col in mean_ret.columns:
        print(f"  {col:>8}", end="")
    print()
    for stg in mean_ret.index:
        n = int(nobs.loc[stg].iloc[0])
        print(f"  {stg:<6} (n={n:3d})", end="")
        for col in mean_ret.columns:
            m = mean_ret.loc[stg, col]
            t = tstat.loc[stg, col]
            star = "*" if not pd.isna(t) and abs(t) >= 1.5 else " "
            val = f"{m:+.2f}{star}" if not pd.isna(m) else "    NaN"
            print(f"  {val:>8}", end="")
        print()

    common = stage.index.intersection(factors.index)
    lo, hi = str(common[0])[:7], str(common[-1])[:7]

    plot_stage_heatmap(
        mean_ret=mean_ret,
        tstat=tstat,
        nobs=nobs,
        title=f"JP Business Cycle × Factor Returns\n({lo} – {hi}, n per stage shown in parentheses)",
        output_path=ROOT / "data" / "jp" / "stage_factor_heatmap.png",
    )


# ─────────────────────────────────────────────────────────────────────────────
# US: Sector analysis
# ─────────────────────────────────────────────────────────────────────────────

def run_us() -> None:
    print("\n=== US: ステージ別セクターリターン ===")

    stage = _load_stage("us")
    print(f"  Stage series: {_sample_label(stage.index)}  ({len(stage)} obs)")

    # 12 Industry Portfolios (VW monthly)
    ind12 = french.fetch(
        "12_Industry_Portfolios_CSV.zip",
        section_keyword="Average Value Weighted Returns -- Monthly",
    )
    print(f"  US 12 Ind:    {_sample_label(ind12.index)}  ({len(ind12)} obs)  cols={list(ind12.columns)}")

    # US 5-Factor for RF
    ff5_us = french.fetch("F-F_Research_Data_5_Factors_2x3_CSV.zip")
    rf = ff5_us["RF"].rename("RF")
    print(f"  US RF:        {_sample_label(rf.index)}  ({len(rf)} obs)")

    mean_ret, tstat, nobs = compute_stage_stats(stage, ind12, rf_series=rf)

    # Print summary
    print("\n  Mean monthly excess return (%) by stage:")
    print(f"  {'Stage':<6}", end="")
    for col in mean_ret.columns:
        print(f"  {col:>7}", end="")
    print()
    for stg in mean_ret.index:
        n = int(nobs.loc[stg].iloc[0])
        print(f"  {stg:<6} (n={n:3d})", end="")
        for col in mean_ret.columns:
            m = mean_ret.loc[stg, col]
            t = tstat.loc[stg, col]
            star = "*" if not pd.isna(t) and abs(t) >= 1.5 else " "
            val = f"{m:+.2f}{star}" if not pd.isna(m) else "   NaN"
            print(f"  {val:>7}", end="")
        print()

    common = stage.index.intersection(ind12.index)
    lo, hi = str(common[0])[:7], str(common[-1])[:7]

    plot_stage_heatmap(
        mean_ret=mean_ret,
        tstat=tstat,
        nobs=nobs,
        title=f"US Business Cycle × Sector Returns (12 Industry)\n({lo} – {hi}, VW excess return vs RF)",
        output_path=ROOT / "data" / "us" / "stage_sector_heatmap.png",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analysis A: ステージ別リターン")
    parser.add_argument(
        "--country", choices=["jp", "us"], default=None,
        help="jp または us。省略時は両方実行。",
    )
    args = parser.parse_args()

    if args.country is None or args.country == "jp":
        run_jp()

    if args.country is None or args.country == "us":
        run_us()
