"""
正規化レイヤー: 定常化 → Z-score → パーセンタイル変換。

全関数は pd.Series を入力/出力とし、副作用を持たない。
ローリング窓のデフォルトは120ヶ月(10年)。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# 定常化変換
# ---------------------------------------------------------------------

def to_yoy_pct(s: pd.Series, periods: int = 12) -> pd.Series:
    """前年同月比(%)。月次データ前提なので periods=12。"""
    return s.pct_change(periods=periods) * 100.0


def to_mom_pct(s: pd.Series, periods: int = 1) -> pd.Series:
    """前月比(%)。"""
    return s.pct_change(periods=periods) * 100.0


def to_diff(s: pd.Series, periods: int = 1) -> pd.Series:
    """差分(レベル指標を定常化したい時に使う)。"""
    return s.diff(periods=periods)


def apply_transform(s: pd.Series, transform: str) -> pd.Series:
    """yaml の `transform` フィールドからディスパッチ。"""
    if transform == "level":
        return s
    if transform == "yoy_pct":
        return to_yoy_pct(s)
    if transform == "mom_pct":
        return to_mom_pct(s)
    if transform == "diff":
        return to_diff(s)
    if transform == "derived":
        # 派生指標は loader 側で formula を評価する。ここではそのまま返す
        return s
    raise ValueError(f"Unknown transform: {transform}")


# ---------------------------------------------------------------------
# 外れ値処理
# ---------------------------------------------------------------------

def winsorize(
    s: pd.Series,
    lower: float = 0.01,
    upper: float = 0.99,
) -> pd.Series:
    """全期間の分位点で外れ値をクリップ。

    COVID-19 のような構造ショックで Z スコアが歪むのを防ぐ。
    """
    lo, hi = s.quantile(lower), s.quantile(upper)
    return s.clip(lower=lo, upper=hi)


def winsorize_rolling(
    s: pd.Series,
    window: int = 120,
    min_periods: int = 60,
    lower: float = 0.01,
    upper: float = 0.99,
) -> pd.Series:
    """ローリング窓内での外れ値クリップ(構造変化に追従)。"""
    lo = s.rolling(window=window, min_periods=min_periods).quantile(lower)
    hi = s.rolling(window=window, min_periods=min_periods).quantile(upper)
    return s.clip(lower=lo, upper=hi)


# ---------------------------------------------------------------------
# 標準化
# ---------------------------------------------------------------------

def rolling_zscore(
    s: pd.Series,
    window: int = 120,
    min_periods: int = 60,
) -> pd.Series:
    """ローリング Z-score。デフォルト10年窓、最小5年。"""
    mean = s.rolling(window=window, min_periods=min_periods).mean()
    std = s.rolling(window=window, min_periods=min_periods).std()
    return (s - mean) / std


def rolling_percentile(
    s: pd.Series,
    window: int = 120,
    min_periods: int = 60,
) -> pd.Series:
    """ローリング窓内でのパーセンタイルランク(0.0-1.0)。

    現在値が窓内で何番目に位置するかを返す。
    分布の裾に対する頑健性で Z-score より優れる。
    """
    def _last_rank(x: pd.Series) -> float:
        if len(x) < 2:
            return np.nan
        # 現在値の rank をパーセンタイルで返す
        return x.rank(pct=True).iloc[-1]

    return s.rolling(window=window, min_periods=min_periods).apply(
        _last_rank, raw=False
    )
