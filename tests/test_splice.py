"""
composite_sources.fetch_splice のユニットテスト。

get_adapter をモックして FRED / e-Stat への実 API 呼び出しなしで実行。
"""
from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from bcycle_jp.core.composite_sources import fetch_splice


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _series(data: dict[str, float], name: str = "s") -> pd.Series:
    idx = pd.DatetimeIndex([pd.Timestamp(d) for d in data])
    return pd.Series(list(data.values()), index=idx, name=name)


def _mock_adapter(series: pd.Series, available: bool = True) -> MagicMock:
    m = MagicMock()
    m.is_available.return_value = available
    m.fetch.return_value = series
    return m


# ---------------------------------------------------------------------------
# 基本的な結合
# ---------------------------------------------------------------------------


class TestFetchSpliceBasic:
    def _sources(self, fred_end="2017-12-31", estat_start="2018-01-01"):
        return {
            "fred": {
                "series_id": "JPNPROINDMISMEI",
                "end_date": fred_end,
            },
            "estat": {
                "stats_data_id": "0004015800",
                "start_date": estat_start,
            },
        }

    def _make_adapters(self):
        fred_data = _series({
            "1985-01-01": 70.0,
            "1985-02-01": 71.0,
            "2017-11-01": 99.0,
            "2017-12-01": 100.0,
        }, name="fred")
        estat_data = _series({
            "2018-01-01": 112.3,
            "2018-02-01": 114.6,
            "2025-01-01": 102.0,
        }, name="estat")
        return _mock_adapter(fred_data), _mock_adapter(estat_data)

    @patch("bcycle_jp.core.composite_sources.get_adapter")
    def test_concatenates_two_sources(self, mock_get):
        fred_adapter, estat_adapter = self._make_adapters()
        mock_get.side_effect = lambda n: fred_adapter if n == "fred" else estat_adapter

        result = fetch_splice(self._sources(), global_start=date(1985, 1, 1))

        assert isinstance(result, pd.Series)
        assert pd.Timestamp("1985-01-01") in result.index
        assert pd.Timestamp("2017-12-01") in result.index
        assert pd.Timestamp("2018-01-01") in result.index
        assert pd.Timestamp("2025-01-01") in result.index

    @patch("bcycle_jp.core.composite_sources.get_adapter")
    def test_result_is_sorted(self, mock_get):
        fred_adapter, estat_adapter = self._make_adapters()
        mock_get.side_effect = lambda n: fred_adapter if n == "fred" else estat_adapter

        result = fetch_splice(self._sources(), global_start=date(1985, 1, 1))

        assert result.index.is_monotonic_increasing

    @patch("bcycle_jp.core.composite_sources.get_adapter")
    def test_fred_called_with_end_date(self, mock_get):
        fred_adapter, estat_adapter = self._make_adapters()
        mock_get.side_effect = lambda n: fred_adapter if n == "fred" else estat_adapter

        fetch_splice(self._sources(), global_start=date(1985, 1, 1))

        _, kwargs = fred_adapter.fetch.call_args
        assert kwargs["end"] == date(2017, 12, 31)

    @patch("bcycle_jp.core.composite_sources.get_adapter")
    def test_estat_called_with_start_date(self, mock_get):
        fred_adapter, estat_adapter = self._make_adapters()
        mock_get.side_effect = lambda n: fred_adapter if n == "fred" else estat_adapter

        fetch_splice(self._sources(), global_start=date(1985, 1, 1))

        _, kwargs = estat_adapter.fetch.call_args
        assert kwargs["start"] == date(2018, 1, 1)

    @patch("bcycle_jp.core.composite_sources.get_adapter")
    def test_splice_meta_keys_stripped_from_adapter_config(self, mock_get):
        fred_adapter, estat_adapter = self._make_adapters()
        mock_get.side_effect = lambda n: fred_adapter if n == "fred" else estat_adapter

        fetch_splice(self._sources(), global_start=date(1985, 1, 1))

        fred_config = fred_adapter.fetch.call_args[0][0]
        estat_config = estat_adapter.fetch.call_args[0][0]
        assert "end_date" not in fred_config
        assert "start_date" not in estat_config


# ---------------------------------------------------------------------------
# 重複区間: e-Stat(後列挙)が優先
# ---------------------------------------------------------------------------


class TestFetchSpliceOverlap:
    @patch("bcycle_jp.core.composite_sources.get_adapter")
    def test_later_source_wins_on_overlap(self, mock_get):
        # FRED と e-Stat が 2018-01 に重複する設定
        fred_data = _series({
            "2017-12-01": 100.0,
            "2018-01-01": 99.0,   # ← FRED の値
            "2018-02-01": 98.0,
        })
        estat_data = _series({
            "2018-01-01": 112.3,  # ← e-Stat の値(こちらが優先されるべき)
            "2018-02-01": 114.6,
        })
        fred_a = _mock_adapter(fred_data)
        estat_a = _mock_adapter(estat_data)
        mock_get.side_effect = lambda n: fred_a if n == "fred" else estat_a

        sources = {
            "fred": {"series_id": "X", "end_date": "2018-02-28"},
            "estat": {"stats_data_id": "Y", "start_date": "2018-01-01"},
        }
        result = fetch_splice(sources, global_start=date(2017, 1, 1))

        # 2018-01 は e-Stat の値
        assert result.loc[pd.Timestamp("2018-01-01")] == pytest.approx(112.3)
        assert result.loc[pd.Timestamp("2018-02-01")] == pytest.approx(114.6)
        # 2017-12 は FRED のみ
        assert result.loc[pd.Timestamp("2017-12-01")] == pytest.approx(100.0)

    @patch("bcycle_jp.core.composite_sources.get_adapter")
    def test_no_duplicate_index_after_overlap(self, mock_get):
        fred_data = _series({"2018-01-01": 99.0, "2018-02-01": 98.0})
        estat_data = _series({"2018-01-01": 112.3, "2018-02-01": 114.6})
        fred_a = _mock_adapter(fred_data)
        estat_a = _mock_adapter(estat_data)
        mock_get.side_effect = lambda n: fred_a if n == "fred" else estat_a

        sources = {
            "fred": {"series_id": "X"},
            "estat": {"stats_data_id": "Y"},
        }
        result = fetch_splice(sources, global_start=date(2018, 1, 1))

        assert result.index.is_unique


# ---------------------------------------------------------------------------
# エラー・フォールバック
# ---------------------------------------------------------------------------


class TestFetchSpliceFallback:
    @patch("bcycle_jp.core.composite_sources.get_adapter")
    def test_skips_unavailable_adapter(self, mock_get):
        fred_a = _mock_adapter(_series({"2018-01-01": 100.0}), available=False)
        estat_a = _mock_adapter(_series({"2018-01-01": 112.3}))
        mock_get.side_effect = lambda n: fred_a if n == "fred" else estat_a

        sources = {"fred": {"series_id": "X"}, "estat": {"stats_data_id": "Y"}}
        result = fetch_splice(sources, global_start=date(2018, 1, 1))

        assert len(result) == 1
        assert result.iloc[0] == pytest.approx(112.3)

    @patch("bcycle_jp.core.composite_sources.get_adapter")
    def test_all_unavailable_raises(self, mock_get):
        mock_get.return_value = _mock_adapter(_series({}), available=False)

        sources = {"fred": {"series_id": "X"}, "estat": {"stats_data_id": "Y"}}
        with pytest.raises(RuntimeError, match="splice"):
            fetch_splice(sources, global_start=date(2018, 1, 1))

    @patch("bcycle_jp.core.composite_sources.get_adapter")
    def test_unknown_source_skipped(self, mock_get):
        mock_get.side_effect = ValueError("未登録")
        estat_a = _mock_adapter(_series({"2018-01-01": 112.3}))

        # bloomberg は未登録なのでスキップ、estat は使える場合を想定
        def side_effect(name):
            if name == "estat":
                return estat_a
            raise ValueError("未登録")

        mock_get.side_effect = side_effect

        sources = {"bloomberg": {"ticker": "Z"}, "estat": {"stats_data_id": "Y"}}
        result = fetch_splice(sources, global_start=date(2018, 1, 1))
        assert len(result) == 1

    @patch("bcycle_jp.core.composite_sources.get_adapter")
    def test_global_end_clamps_source_end_date(self, mock_get):
        fred_a = _mock_adapter(_series({"2017-12-01": 100.0}))
        mock_get.return_value = fred_a

        sources = {"fred": {"series_id": "X", "end_date": "2020-12-31"}}
        fetch_splice(sources, global_start=date(2017, 1, 1), global_end=date(2018, 6, 30))

        _, kwargs = fred_a.fetch.call_args
        # global_end(2018-06) と end_date(2020-12) の min → 2018-06
        assert kwargs["end"] == date(2018, 6, 30)
