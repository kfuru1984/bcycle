"""
ステージ別リターン分析。

data/{country}/level_momentum.parquet のステージ履歴と
Ken French リターン系列を月次でジョインし、
ステージ × アセット の統計量(平均超過リターン・t値・観測数)を計算する。
ヒートマップを PNG に出力する。
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.font_manager as _fm
import numpy as np
import pandas as pd
from scipy import stats

# Japanese font: pick first candidate whose file is actually on disk
_JP_FONT_CANDIDATES = ["Noto Sans JP", "Yu Gothic", "Meiryo", "MS Gothic", "MS PGothic"]
_jp_font: str | None = None
_available_names = {_fm.FontProperties(fname=f).get_name()
                    for f in _fm.findSystemFonts(fontext="ttf")}
for _cand in _JP_FONT_CANDIDATES:
    if _cand in _available_names:
        _jp_font = _cand
        break

STAGES = ["回復", "上昇", "成熟", "軟化", "下降"]

_STAGE_ROW_COLORS = {
    "回復": "#fffacd",
    "上昇": "#90ee90",
    "成熟": "#add8e6",
    "軟化": "#ffd700",
    "下降": "#ffb6c1",
}


# ─────────────────────────────────────────────────────────────────────────────
# Statistics
# ─────────────────────────────────────────────────────────────────────────────

def compute_stage_stats(
    stage_series: pd.Series,
    returns_df: pd.DataFrame,
    rf_series: pd.Series | None = None,
    min_obs: int = 12,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Compute per-stage return statistics.

    Parameters
    ----------
    stage_series : monthly stage labels (index = DatetimeIndex)
    returns_df   : monthly returns in % (index = DatetimeIndex)
    rf_series    : monthly risk-free rate (%). If given, excess = returns - RF.
                   For factor data (already excess), pass None.
    min_obs      : minimum observations required; else NaN

    Returns
    -------
    mean_ret : DataFrame[stage × asset]  — mean excess return (%)
    tstat    : DataFrame[stage × asset]  — one-sample t-stat vs 0
    nobs     : DataFrame[stage × asset]  — observation count
    """
    common = stage_series.index.intersection(returns_df.index)
    stage = stage_series.loc[common]
    rets  = returns_df.loc[common].copy()

    if rf_series is not None:
        rf_aligned = rf_series.reindex(common).ffill()
        rets = rets.subtract(rf_aligned, axis=0)

    mean_rows, tstat_rows, nobs_rows = {}, {}, {}

    for stg in STAGES:
        mask = stage == stg
        sub  = rets.loc[mask]
        n    = int(mask.sum())

        nobs_rows[stg] = pd.Series(n, index=rets.columns)

        if n < min_obs:
            mean_rows[stg]  = pd.Series(float("nan"), index=rets.columns)
            tstat_rows[stg] = pd.Series(float("nan"), index=rets.columns)
            continue

        mean_rows[stg] = sub.mean()

        ts_vals: dict[str, float] = {}
        for col in rets.columns:
            vals = sub[col].dropna()
            if len(vals) >= min_obs:
                t, _ = stats.ttest_1samp(vals, 0.0)
                ts_vals[col] = float(t)
            else:
                ts_vals[col] = float("nan")
        tstat_rows[stg] = pd.Series(ts_vals)

    mean_ret = pd.DataFrame(mean_rows).T.reindex(STAGES)
    tstat    = pd.DataFrame(tstat_rows).T.reindex(STAGES)
    nobs     = pd.DataFrame(nobs_rows).T.reindex(STAGES)

    return mean_ret, tstat, nobs


# ─────────────────────────────────────────────────────────────────────────────
# Heatmap
# ─────────────────────────────────────────────────────────────────────────────

def plot_stage_heatmap(
    mean_ret: pd.DataFrame,
    tstat: pd.DataFrame,
    nobs: pd.DataFrame,
    title: str,
    output_path: Path,
    t_threshold: float = 1.5,
) -> None:
    """
    Draw stage × asset heatmap.

    Cells with |t| >= t_threshold are colored (RdYlGn centered at 0).
    All cells show annotation: mean return (%) on top, t-value below.
    Cells below threshold have white background.
    """
    if _jp_font:
        plt.rcParams["font.family"] = _jp_font

    n_stages = len(STAGES)
    n_assets  = mean_ret.shape[1]
    col_labels = list(mean_ret.columns)

    fig_w = max(8.0, n_assets * 1.1 + 2.5)
    fig_h = n_stages * 0.95 + 2.2
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # color range: symmetric around 0, based on significant cells
    sig_mask = tstat.abs() >= t_threshold
    sig_vals = mean_ret.where(sig_mask).values.ravel()
    sig_vals = sig_vals[~np.isnan(sig_vals)]
    vmax = float(np.abs(sig_vals).max()) if len(sig_vals) else 1.0
    vmax = max(vmax, 0.5)

    cmap = plt.cm.RdYlGn
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    for i, stg in enumerate(STAGES):
        row_i = n_stages - 1 - i  # top stage at top of plot
        for j in range(n_assets):
            m   = mean_ret.loc[stg].iloc[j]
            t   = tstat.loc[stg].iloc[j]
            sig = (not pd.isna(t)) and (abs(t) >= t_threshold)

            # background
            if sig and not pd.isna(m):
                face = cmap(norm(m))
            else:
                face = "white"

            rect = plt.Rectangle(
                [j, row_i], 1, 1,
                facecolor=face,
                edgecolor="lightgray",
                linewidth=0.6,
            )
            ax.add_patch(rect)

            # text annotation
            if not pd.isna(m):
                mean_txt  = f"{m:+.2f}%"
                t_txt     = f"t={t:+.1f}" if not pd.isna(t) else ""
                cell_text = f"{mean_txt}\n{t_txt}" if t_txt else mean_txt

                # contrast: dark face → white text
                brightness = 0.5
                if sig and not pd.isna(m):
                    r, g, b, _ = cmap(norm(m))
                    brightness = 0.299 * r + 0.587 * g + 0.114 * b
                txt_color = "white" if brightness < 0.45 else "black"

                ax.text(
                    j + 0.5, row_i + 0.5, cell_text,
                    ha="center", va="center",
                    fontsize=7.5, color=txt_color,
                    linespacing=1.3,
                )

    # axes setup
    ax.set_xlim(0, n_assets)
    ax.set_ylim(0, n_stages)
    ax.set_xticks(np.arange(n_assets) + 0.5)
    ax.set_xticklabels(col_labels, rotation=40, ha="right", fontsize=9)
    ax.set_yticks(np.arange(n_stages) + 0.5)
    ax.set_yticklabels(list(reversed(STAGES)), fontsize=10)
    ax.tick_params(length=0)

    # stage-label color strips on left margin
    for i, stg in enumerate(STAGES):
        row_i = n_stages - 1 - i
        ax.add_patch(plt.Rectangle(
            [-0.25, row_i], 0.22, 1,
            facecolor=_STAGE_ROW_COLORS.get(stg, "#f0f0f0"),
            clip_on=False,
        ))

    # colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.65, pad=0.02)
    cbar.set_label("Mean Excess Return (%/mo)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    ax.set_title(title, fontsize=11, fontweight="bold", pad=10)

    # footnote
    fig.text(
        0.5, 0.01,
        f"Color shown only where |t| >= {t_threshold:.1f}  |  white cells: insufficient significance",
        ha="center", fontsize=7, color="gray",
    )

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"出力: {output_path}")
