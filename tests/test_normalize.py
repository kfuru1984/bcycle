"""normalize.py のユニットテスト。

純ロジック層なので、合成データで挙動を固める。
Claude Code でアダプタ実装した後の安全網としても機能する。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bcycle_jp.core.normalize import (
    apply_transform,
    rolling_percentile,
    rolling_zscore,
    to_yoy_pct,
    winsorize,
)


@pytest.fixture
def monthly_series() -> pd.Series:
    """15年分の月次合成データ(トレンド + サイクル + ノイズ)。"""
    rng = pd.date_range("2010-01-01", periods=180, freq="MS")
    np.random.seed(42)
    trend = np.linspace(100, 130, 180)
    cycle = 5 * np.sin(np.arange(180) * 2 * np.pi / 60)  # 5年サイクル
    noise = np.random.normal(0, 1, 180)
    return pd.Series(trend + cycle + noise, index=rng, name="test")


def test_yoy_pct_returns_same_length(monthly_series):
    yoy = to_yoy_pct(monthly_series)
    assert len(yoy) == len(monthly_series)
    # 最初の12ヶ月はNaN
    assert yoy.iloc[:12].isna().all()
    # 13ヶ月目以降は有限値
    assert yoy.iloc[12:].notna().all()


def test_rolling_zscore_centered_around_zero():
    """定常データなら rolling Z は0中心になる。

    トレンド込みのデータだと窓内平均が現値に追従できず Z が偏るので、
    Z スコアの中心性は「定常化後」の系列に対して期待すべき性質。
    """
    rng = pd.date_range("2010-01-01", periods=180, freq="MS")
    np.random.seed(0)
    # 定常データ(平均100、振幅5のサイクル + ノイズ)
    cycle = 5 * np.sin(np.arange(180) * 2 * np.pi / 60)
    noise = np.random.normal(0, 1, 180)
    s = pd.Series(100 + cycle + noise, index=rng)

    z = rolling_zscore(s, window=60, min_periods=30)
    stable = z.dropna()
    assert abs(stable.mean()) < 0.4
    assert 0.7 < stable.std() < 1.5


def test_rolling_percentile_in_unit_interval(monthly_series):
    pct = rolling_percentile(monthly_series, window=60, min_periods=30)
    stable = pct.dropna()
    assert (stable >= 0).all()
    assert (stable <= 1).all()


def test_winsorize_caps_extremes():
    s = pd.Series([1, 2, 3, 4, 5, 100, -100])
    w = winsorize(s, lower=0.1, upper=0.9)
    assert w.max() < 100
    assert w.min() > -100


def test_apply_transform_dispatch():
    s = pd.Series([100, 105, 110, 108], index=pd.date_range("2024-01", periods=4, freq="MS"))
    # level はそのまま
    assert (apply_transform(s, "level") == s).all()
    # 未知の transform はエラー
    with pytest.raises(ValueError):
        apply_transform(s, "unknown_transform")
