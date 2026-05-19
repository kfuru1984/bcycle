"""
ステージ判定: レベル × モメンタム のクアドラント + ヒステリシス。

5ステージ:
  回復 (Recovery)   レベル低 × モメンタム+
  上昇 (Expansion)  レベル中 × モメンタム+
  成熟 (Mature)     レベル高 × モメンタム+
  軟化 (Softening)  レベル高 × モメンタム-
  下降 (Decline)    レベル低-中 × モメンタム-
"""
from __future__ import annotations

import pandas as pd

STAGES = ["回復", "上昇", "成熟", "軟化", "下降"]


def _raw_stage(level: float, momentum: float,
               level_low: float, level_high: float,
               momentum_threshold: float) -> str | None:
    """境界判定のみ。NaN は None を返す。"""
    if pd.isna(level) or pd.isna(momentum):
        return None
    if momentum >= momentum_threshold:
        if level < level_low:
            return "回復"
        if level < level_high:
            return "上昇"
        return "成熟"
    else:
        if level >= level_high:
            return "軟化"
        return "下降"


def classify_stage(
    level: pd.Series,
    momentum: pd.Series,
    level_low: float = 33.0,
    level_high: float = 67.0,
    momentum_threshold: float = 0.0,
    hysteresis_periods: int = 2,
) -> pd.Series:
    """ステージ判定 with ヒステリシス。

    `hysteresis_periods` 期連続で同じ raw_stage が続かないと遷移しない。
    境界フリップフロップを防ぐ。
    """
    raw = pd.Series(
        [
            _raw_stage(l, m, level_low, level_high, momentum_threshold)
            for l, m in zip(level, momentum)
        ],
        index=level.index,
    )

    if hysteresis_periods <= 1:
        return raw

    smoothed: list[str | None] = []
    current: str | None = None
    candidate: str | None = None
    candidate_count = 0

    for v in raw:
        if v is None:
            smoothed.append(current)
            candidate = None
            candidate_count = 0
            continue

        if current is None:
            # 初期化: 最初の非None値で確定
            current = v
            smoothed.append(current)
            candidate = None
            candidate_count = 0
            continue

        if v == current:
            smoothed.append(current)
            candidate = None
            candidate_count = 0
        else:
            # 遷移候補の蓄積
            if v == candidate:
                candidate_count += 1
            else:
                candidate = v
                candidate_count = 1

            if candidate_count >= hysteresis_periods:
                current = candidate
                candidate = None
                candidate_count = 0

            smoothed.append(current)

    return pd.Series(smoothed, index=level.index, name="stage")


def stage_confidence(
    indicator_z_scores: pd.DataFrame,
    stage: pd.Series,
) -> pd.Series:
    """ステージ判定の確度を 0-1 で返す。

    全指標の Z スコア符号がステージと整合する割合を計算する簡易版。
    例: ステージが「下降」なら各指標が負方向に振れている割合。
    """
    sign_expected = {
        "回復": +1, "上昇": +1, "成熟": +1, "軟化": -1, "下降": -1,
    }
    conf: list[float] = []
    for t, stg in zip(indicator_z_scores.index, stage):
        if stg is None or pd.isna(stg):
            conf.append(float("nan"))
            continue
        row = indicator_z_scores.loc[t].dropna()
        if len(row) == 0:
            conf.append(float("nan"))
            continue
        expected = sign_expected.get(stg, 0)
        if expected == 0:
            conf.append(float("nan"))
            continue
        agreed = ((row * expected) > 0).sum()
        conf.append(agreed / len(row))
    return pd.Series(conf, index=indicator_z_scores.index, name="confidence")
