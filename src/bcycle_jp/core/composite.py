"""
合成スコア: サイクルレベル(0-100)とモメンタム(z-score)を算出。

設計:
  - レベル: 各指標のパーセンタイル平均 × 100、または z 加重平均をパーセンタイル化
  - モメンタム: レベルの 3M 変化を Z 化
  - 重みは settings.yaml で外部化(MVP は等加重スタート)
"""
from __future__ import annotations

import pandas as pd

from .normalize import rolling_percentile, rolling_zscore


def compute_level(
    indicators: dict[str, pd.Series],
    weights: dict[str, float] | None = None,
    method: str = "percentile_mean",
    window: int = 120,
    min_periods: int = 60,
    min_indicator_frac: float = 0.5,
) -> pd.Series:
    """指標群から 0-100 のサイクルレベルを合成。

    Parameters
    ----------
    indicators : dict[str, pd.Series]
        {indicator_id: 既に定常化・(必要なら符号反転)済の系列}
        ※失業率のような「高い=悪い」系は呼び出し側で符号反転して渡す
    weights : dict[str, float] | None
        各指標の重み。None なら等加重。
    method : {"percentile_mean", "z_mean"}
        - "percentile_mean": 各指標のローリングパーセンタイル平均 × 100
        - "z_mean": Z 加重平均をパーセンタイル化して × 100
    min_indicator_frac : float
        有効指標数がこの割合未満の月は NaN にする (デフォルト 0.5 = 50%)。
        weight > 0 の指標のみカウント対象。

    Returns
    -------
    pd.Series
        DatetimeIndex で 0-100 のレベルスコア
    """
    df = pd.DataFrame(indicators)

    if weights is None:
        weights = {k: 1.0 for k in df.columns}
    w = pd.Series(weights).reindex(df.columns).fillna(0.0)
    if w.sum() == 0:
        raise ValueError("重みの合計が0。指標が空か weights キー名がミスマッチ")
    w = w / w.sum()

    # weight > 0 の指標の総数 (min_indicator_frac の分母)
    n_active = int((w > 0).sum())

    if method == "percentile_mean":
        pct = df.apply(lambda s: rolling_percentile(s, window, min_periods))
        level = (pct * w).sum(axis=1) / (pct.notna().astype(float) * w).sum(axis=1)
        level = level * 100.0
        # 有効指標数が閾値未満の月を NaN にする
        n_available = (pct.notna() & (w > 0)).sum(axis=1)
        level[n_available < min_indicator_frac * n_active] = float("nan")
        return level

    if method == "z_mean":
        z = df.apply(lambda s: rolling_zscore(s, window, min_periods))
        z_composite = (z * w).sum(axis=1) / (z.notna().astype(float) * w).sum(axis=1)
        level = rolling_percentile(z_composite, window, min_periods) * 100.0
        n_available = (z.notna() & (w > 0)).sum(axis=1)
        level[n_available < min_indicator_frac * n_active] = float("nan")
        return level

    raise ValueError(f"Unknown method: {method}")


def compute_momentum(
    level: pd.Series,
    lookback: int = 3,
    window: int = 120,
    min_periods: int = 60,
) -> pd.Series:
    """レベルの lookback ヶ月変化を Z 化してモメンタムとする。

    +0.5 以上で「明確な上向き」、-0.5 以下で「明確な下向き」、
    その間はノイズ圏という解釈が標準的。
    """
    change = level.diff(periods=lookback)
    return rolling_zscore(change, window=window, min_periods=min_periods)
