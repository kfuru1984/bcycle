"""
FRED アダプタのユニットテスト。

fredapi.Fred をモックして API キー不要で実行できる。
"""
from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from bcycle_jp.adapters.fred import FredAdapter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fred_series(data: dict[str, float]) -> pd.Series:
    """fredapi.Fred.get_series が返す形式のモックシリーズを生成する。"""
    idx = pd.DatetimeIndex(list(data.keys()))
    return pd.Series(list(data.values()), index=idx, name="TEST")


_SAMPLE_DATA = {
    "1985-01-31": 4.9,
    "1985-02-28": 5.0,
    "1985-03-31": 5.1,
    "1985-04-30": 5.2,
}


# ---------------------------------------------------------------------------
# FredAdapter
# ---------------------------------------------------------------------------


class TestFredAdapter:
    def _adapter(self) -> FredAdapter:
        return FredAdapter(api_key="dummy_key_32chars_xxxxxxxxxxxxxxx")

    def test_is_available_with_key(self):
        assert self._adapter().is_available() is True

    def test_is_available_without_key(self, monkeypatch):
        monkeypatch.delenv("FRED_API_KEY", raising=False)
        adapter = FredAdapter(api_key=None)
        adapter.api_key = None  # env から取得した値も無効化
        assert adapter.is_available() is False

    def test_unavailable_raises(self, monkeypatch):
        monkeypatch.delenv("FRED_API_KEY", raising=False)
        adapter = FredAdapter(api_key=None)
        adapter.api_key = None
        with pytest.raises(RuntimeError, match="FRED_API_KEY"):
            adapter.fetch({"series_id": "TEST"}, start=date(1985, 1, 1))

    def test_missing_series_id_raises(self):
        with pytest.raises(ValueError, match="series_id"):
            self._adapter().fetch({}, start=date(1985, 1, 1))

    @patch("bcycle_jp.adapters.fred.Fred")
    def test_fetch_returns_month_start_index(self, mock_fred_cls, tmp_path, monkeypatch):
        monkeypatch.setattr("bcycle_jp.adapters.fred._CACHE_DIR", tmp_path)
        mock_fred = MagicMock()
        mock_fred_cls.return_value = mock_fred
        mock_fred.get_series.return_value = _make_fred_series(_SAMPLE_DATA)

        result = self._adapter().fetch({"series_id": "TEST"}, start=date(1985, 1, 1))

        assert isinstance(result, pd.Series)
        # 月末日 → 月初に正規化されているか
        assert result.index[0] == pd.Timestamp("1985-01-01")
        assert result.index[-1] == pd.Timestamp("1985-04-01")
        assert len(result) == 4

    @patch("bcycle_jp.adapters.fred.Fred")
    def test_fetch_drops_nan(self, mock_fred_cls, tmp_path, monkeypatch):
        monkeypatch.setattr("bcycle_jp.adapters.fred._CACHE_DIR", tmp_path)
        mock_fred = MagicMock()
        mock_fred_cls.return_value = mock_fred
        import numpy as np
        data_with_nan = _make_fred_series(_SAMPLE_DATA).copy()
        data_with_nan.iloc[1] = float("nan")
        mock_fred.get_series.return_value = data_with_nan

        result = self._adapter().fetch({"series_id": "TEST"}, start=date(1985, 1, 1))
        assert len(result) == 3  # NaN 除外

    @patch("bcycle_jp.adapters.fred.Fred")
    def test_fetch_uses_cache(self, mock_fred_cls, tmp_path, monkeypatch):
        monkeypatch.setattr("bcycle_jp.adapters.fred._CACHE_DIR", tmp_path)
        mock_fred = MagicMock()
        mock_fred_cls.return_value = mock_fred
        mock_fred.get_series.return_value = _make_fred_series(_SAMPLE_DATA)

        adapter = self._adapter()
        adapter.fetch({"series_id": "TEST"}, start=date(1985, 1, 1))
        assert mock_fred.get_series.call_count == 1

        # 2回目はキャッシュから
        adapter.fetch({"series_id": "TEST"}, start=date(1985, 1, 1))
        assert mock_fred.get_series.call_count == 1

    @patch("bcycle_jp.adapters.fred.Fred")
    def test_date_filter_start(self, mock_fred_cls, tmp_path, monkeypatch):
        monkeypatch.setattr("bcycle_jp.adapters.fred._CACHE_DIR", tmp_path)
        mock_fred = MagicMock()
        mock_fred_cls.return_value = mock_fred
        mock_fred.get_series.return_value = _make_fred_series(_SAMPLE_DATA)

        result = self._adapter().fetch({"series_id": "TEST"}, start=date(1985, 3, 1))
        assert result.index[0] == pd.Timestamp("1985-03-01")
        assert len(result) == 2

    @patch("bcycle_jp.adapters.fred.Fred")
    def test_date_filter_end(self, mock_fred_cls, tmp_path, monkeypatch):
        monkeypatch.setattr("bcycle_jp.adapters.fred._CACHE_DIR", tmp_path)
        mock_fred = MagicMock()
        mock_fred_cls.return_value = mock_fred
        mock_fred.get_series.return_value = _make_fred_series(_SAMPLE_DATA)

        result = self._adapter().fetch(
            {"series_id": "TEST"},
            start=date(1985, 1, 1),
            end=date(1985, 2, 28),
        )
        assert len(result) == 2
        assert result.index[-1] == pd.Timestamp("1985-02-01")

    @patch("bcycle_jp.adapters.fred.Fred")
    def test_different_series_ids_cached_separately(self, mock_fred_cls, tmp_path, monkeypatch):
        monkeypatch.setattr("bcycle_jp.adapters.fred._CACHE_DIR", tmp_path)
        mock_fred = MagicMock()
        mock_fred_cls.return_value = mock_fred
        mock_fred.get_series.return_value = _make_fred_series(_SAMPLE_DATA)

        adapter = self._adapter()
        adapter.fetch({"series_id": "SERIES_A"}, start=date(1985, 1, 1))
        adapter.fetch({"series_id": "SERIES_B"}, start=date(1985, 1, 1))

        cache_files = list(tmp_path.glob("fred_*.parquet"))
        assert len(cache_files) == 2
