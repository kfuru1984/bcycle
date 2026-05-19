"""
FRED (Federal Reserve Economic Data) API アダプタ。

fredapi パッケージを使用。
キャッシュ: data/fred_{series_id}.parquet
"""
from __future__ import annotations

import os
from datetime import date
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from fredapi import Fred

from .base import BaseAdapter

load_dotenv()

_CACHE_DIR = Path(__file__).resolve().parents[3] / "data"
_CACHE_DIR.mkdir(exist_ok=True)


class FredAdapter(BaseAdapter):
    """FRED REST API アダプタ (fredapi ラッパー)。"""

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("FRED_API_KEY")

    def is_available(self) -> bool:
        return bool(self.api_key)

    def fetch(
        self,
        source_config: dict,
        start: date,
        end: date | None = None,
    ) -> pd.Series:
        """FRED から系列を取得して月初 DatetimeIndex の pd.Series で返す。

        source_config の必須フィールド:
          series_id : str   FRED の系列ID (例: "LRUN64TTJPM156S")
        """
        if not self.is_available():
            raise RuntimeError("FRED_API_KEY が未設定 (.env を確認)")

        series_id = source_config.get("series_id")
        if not series_id:
            raise ValueError("source_config に series_id が必要")

        cache_path = _CACHE_DIR / f"fred_{series_id}.parquet"

        if cache_path.exists():
            series = pd.read_parquet(cache_path).squeeze()
            series.index = pd.DatetimeIndex(series.index)
            return self._filter_dates(series, start, end)

        fred = Fred(api_key=self.api_key)
        # キャッシュには全期間を保存する。start/end はキャッシュ後にフィルタ。
        # observation_start/end を絞ると切り詰めデータがキャッシュされるバグを防ぐ。
        raw: pd.Series = fred.get_series(series_id)

        # FRED は日次/月末日など様々な日付を返すため月初に集約
        # 日次系列(T10Y2Yなど)は月平均に変換してから保存する
        raw = raw.dropna()
        raw.index = pd.DatetimeIndex(raw.index)
        raw = raw.resample("MS").mean().dropna()
        raw = raw.sort_index()
        raw.name = series_id

        raw.to_frame("value").to_parquet(cache_path)
        return self._filter_dates(raw, start, end)

    @staticmethod
    def _filter_dates(
        series: pd.Series, start: date, end: date | None
    ) -> pd.Series:
        lo = pd.Timestamp(start)
        hi = pd.Timestamp(end) if end else None
        return series.loc[lo:hi] if hi else series.loc[lo:]
