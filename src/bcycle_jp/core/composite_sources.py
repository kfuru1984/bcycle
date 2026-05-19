"""
複数ソースを日付区間で結合するスプライス合成。

yaml の sources 各エントリに start_date / end_date を追加することで
ソースごとの有効区間を指定できる。重複区間は後に列挙されたソースが優先。

制限事項:
  level 系列をスプライスした後に yoy_pct を適用する場合、
  スプライス境界前後 12 ヶ月は異なる基準年の系列をまたぐため
  YoY 値に誤差が生じる(典型的 < 0.5pp)。
  正確な連続 YoY が必要な場合は end_date / start_date を 13 ヶ月ずらして
  重複区間を設け、e-Stat 優先で上書きすること。
"""
from __future__ import annotations

from datetime import date

import pandas as pd

from bcycle_jp.adapters.registry import get_adapter

_SPLICE_META_KEYS = frozenset({"start_date", "end_date"})


def fetch_splice(
    sources: dict,
    global_start: date,
    global_end: date | None = None,
) -> pd.Series:
    """複数ソースを日付区間で結合し、1本の pd.Series を返す。

    Parameters
    ----------
    sources : dict
        yaml indicators[i].sources 全体。
        各エントリに任意で start_date / end_date (ISO 形式) を含めてよい。
    global_start : date
        全体の取得開始下限。
    global_end : date | None
        全体の取得終了上限。None なら最新まで。

    Returns
    -------
    pd.Series
        月初 DatetimeIndex。重複区間は後に列挙されたソースが優先。
    """
    parts: list[pd.Series] = []

    for src_name, src_config in sources.items():
        try:
            adapter = get_adapter(src_name)
        except ValueError:
            continue  # 未登録ソース (bloomberg など) はスキップ
        if not adapter.is_available():
            continue

        start_str = src_config.get("start_date")
        end_str = src_config.get("end_date")

        effective_start = max(
            global_start,
            date.fromisoformat(start_str) if start_str else global_start,
        )
        raw_end = date.fromisoformat(end_str) if end_str else global_end
        if raw_end and global_end:
            effective_end: date | None = min(raw_end, global_end)
        else:
            effective_end = raw_end if raw_end else global_end

        # アダプタに渡す前にスプライスメタキーを除去
        clean_config = {k: v for k, v in src_config.items() if k not in _SPLICE_META_KEYS}

        try:
            series = adapter.fetch(clean_config, start=effective_start, end=effective_end)
            parts.append(series)
        except Exception:
            continue

    if not parts:
        raise RuntimeError("splice: 利用可能なソースから取得できませんでした")

    # 後に列挙されたソース(parts[-1] 方向)が重複区間で優先
    combined = parts[0]
    for part in parts[1:]:
        overlap_removed = combined[~combined.index.isin(part.index)]
        combined = pd.concat([overlap_removed, part]).sort_index()

    combined.name = "spliced"
    return combined
