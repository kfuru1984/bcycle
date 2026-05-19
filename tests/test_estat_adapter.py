"""
e-Stat アダプタのユニットテスト。

requests.get をモックして API キー不要で実行できる。
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from bcycle_jp.adapters.estat import EStatAdapter, _parse_time_code


# ---------------------------------------------------------------------------
# _parse_time_code
# ---------------------------------------------------------------------------


class TestParseTimeCode:
    def test_6digit_monthly(self):
        ts = _parse_time_code("202401")
        assert ts == pd.Timestamp("2024-01-01")

    def test_6digit_december(self):
        ts = _parse_time_code("202312")
        assert ts == pd.Timestamp("2023-12-01")

    def test_8digit_date(self):
        ts = _parse_time_code("20240115")
        assert ts == pd.Timestamp("2024-01-01")  # 月初に丸める

    def test_10digit_old_format(self):
        ts = _parse_time_code("2024000101")
        assert ts == pd.Timestamp("2024-01-01")

    def test_10digit_month_10(self):
        ts = _parse_time_code("2024001001")
        assert ts == pd.Timestamp("2024-10-01")

    def test_10digit_annual_raises(self):
        with pytest.raises(ValueError, match="年次"):
            _parse_time_code("2024000001")

    def test_unknown_length_raises(self):
        with pytest.raises(ValueError, match="不明"):
            _parse_time_code("202")


# ---------------------------------------------------------------------------
# Helper to build mock responses
# ---------------------------------------------------------------------------


def _make_list_response(tables: list[dict]) -> MagicMock:
    mock = MagicMock()
    mock.raise_for_status.return_value = None
    mock.json.return_value = {
        "GET_STATS_LIST": {
            "DATALIST_INF": {
                "NUMBER": len(tables),
                "TABLE_INF": tables,
            }
        }
    }
    return mock


def _make_data_response(values: list[dict]) -> MagicMock:
    mock = MagicMock()
    mock.raise_for_status.return_value = None
    mock.json.return_value = {
        "GET_STATS_DATA": {
            "STATISTICAL_DATA": {
                "RESULT_INF": {"TOTAL_NUMBER": len(values)},
                "DATA_INF": {"VALUE": values},
            }
        }
    }
    return mock


# ---------------------------------------------------------------------------
# EStatAdapter.resolve_stats_data_id
# ---------------------------------------------------------------------------

_SAMPLE_TABLES = [
    {"@id": "0003191203", "TITLE": {"@no": "1", "$": "生産指数（季節調整済）"}},
    {"@id": "0003191204", "TITLE": {"@no": "2", "$": "生産指数（原指数）"}},
]


class TestResolveStatsDataId:
    def _adapter(self):
        return EStatAdapter(app_id="dummy_key")

    @patch("bcycle_jp.adapters.estat.requests.get")
    def test_returns_first_when_no_keywords(self, mock_get):
        mock_get.return_value = _make_list_response(_SAMPLE_TABLES)
        result = self._adapter().resolve_stats_data_id("00550010")
        assert result == "0003191203"

    @patch("bcycle_jp.adapters.estat.requests.get")
    def test_keyword_match(self, mock_get):
        mock_get.return_value = _make_list_response(_SAMPLE_TABLES)
        result = self._adapter().resolve_stats_data_id(
            "00550010", ["季節調整済"]
        )
        assert result == "0003191203"

    @patch("bcycle_jp.adapters.estat.requests.get")
    def test_keyword_selects_second_table(self, mock_get):
        mock_get.return_value = _make_list_response(_SAMPLE_TABLES)
        result = self._adapter().resolve_stats_data_id("00550010", ["原指数"])
        assert result == "0003191204"

    @patch("bcycle_jp.adapters.estat.requests.get")
    def test_no_keyword_match_warns_and_returns_first(self, mock_get):
        mock_get.return_value = _make_list_response(_SAMPLE_TABLES)
        with pytest.warns(UserWarning):
            result = self._adapter().resolve_stats_data_id(
                "00550010", ["存在しないキーワード"]
            )
        assert result == "0003191203"

    @patch("bcycle_jp.adapters.estat.requests.get")
    def test_single_dict_table_inf(self, mock_get):
        # getStatsList が1件だけの時は dict で返る場合がある
        mock_get.return_value = _make_list_response(_SAMPLE_TABLES[0])
        result = self._adapter().resolve_stats_data_id("00550010")
        assert result == "0003191203"

    @patch("bcycle_jp.adapters.estat.requests.get")
    def test_empty_table_raises(self, mock_get):
        mock = MagicMock()
        mock.raise_for_status.return_value = None
        mock.json.return_value = {
            "GET_STATS_LIST": {"DATALIST_INF": {"NUMBER": 0, "TABLE_INF": []}}
        }
        mock_get.return_value = mock
        with pytest.raises(ValueError, match="見つかりません"):
            self._adapter().resolve_stats_data_id("99999999")


# ---------------------------------------------------------------------------
# EStatAdapter.fetch
# ---------------------------------------------------------------------------

_SAMPLE_VALUES = [
    {"@time": "198501", "@cat01": "00100", "$": "66.7"},
    {"@time": "198502", "@cat01": "00100", "$": "67.2"},
    {"@time": "198503", "@cat01": "00100", "$": "-"},     # NULL は除外
    {"@time": "198504", "@cat01": "00100", "$": "68.0"},
]


class TestFetch:
    def _adapter(self):
        return EStatAdapter(app_id="dummy_key")

    @patch("bcycle_jp.adapters.estat.requests.get")
    def test_fetch_returns_series(self, mock_get, tmp_path, monkeypatch):
        monkeypatch.setattr("bcycle_jp.adapters.estat._CACHE_DIR", tmp_path)
        mock_get.return_value = _make_data_response(_SAMPLE_VALUES)

        src = {"stats_data_id": "0003191203"}
        result = self._adapter().fetch(src, start=date(1985, 1, 1))

        assert isinstance(result, pd.Series)
        assert len(result) == 3  # NULL("-") を除いた3件
        assert result.iloc[0] == pytest.approx(66.7)
        assert result.index[0] == pd.Timestamp("1985-01-01")

    @patch("bcycle_jp.adapters.estat.requests.get")
    def test_fetch_uses_cache(self, mock_get, tmp_path, monkeypatch):
        monkeypatch.setattr("bcycle_jp.adapters.estat._CACHE_DIR", tmp_path)
        mock_get.return_value = _make_data_response(_SAMPLE_VALUES)

        src = {"stats_data_id": "0003191203"}
        adapter = self._adapter()
        # 1回目: API 呼び出し
        adapter.fetch(src, start=date(1985, 1, 1))
        assert mock_get.call_count == 1

        # 2回目: キャッシュから
        adapter.fetch(src, start=date(1985, 1, 1))
        assert mock_get.call_count == 1  # 増えない

    @patch("bcycle_jp.adapters.estat.requests.get")
    def test_fetch_resolves_tbd(self, mock_get, tmp_path, monkeypatch):
        monkeypatch.setattr("bcycle_jp.adapters.estat._CACHE_DIR", tmp_path)
        mock_get.side_effect = [
            _make_list_response(_SAMPLE_TABLES),  # getStatsList
            _make_data_response(_SAMPLE_VALUES),   # getStatsData
        ]

        src = {
            "stats_data_id": "TBD",
            "stats_code": "00550010",
            "table_name_contains": ["季節調整済"],
        }
        result = self._adapter().fetch(src, start=date(1985, 1, 1))
        assert len(result) == 3
        assert mock_get.call_count == 2

    @patch("bcycle_jp.adapters.estat.requests.get")
    def test_date_filter_start(self, mock_get, tmp_path, monkeypatch):
        monkeypatch.setattr("bcycle_jp.adapters.estat._CACHE_DIR", tmp_path)
        mock_get.return_value = _make_data_response(_SAMPLE_VALUES)

        src = {"stats_data_id": "0003191203"}
        result = self._adapter().fetch(src, start=date(1985, 3, 1))
        assert result.index[0] == pd.Timestamp("1985-04-01")

    @patch("bcycle_jp.adapters.estat.requests.get")
    def test_date_filter_end(self, mock_get, tmp_path, monkeypatch):
        monkeypatch.setattr("bcycle_jp.adapters.estat._CACHE_DIR", tmp_path)
        mock_get.return_value = _make_data_response(_SAMPLE_VALUES)

        src = {"stats_data_id": "0003191203"}
        result = self._adapter().fetch(
            src, start=date(1985, 1, 1), end=date(1985, 2, 28)
        )
        assert len(result) == 2
        assert result.index[-1] == pd.Timestamp("1985-02-01")

    @patch("bcycle_jp.adapters.estat.requests.get")
    def test_filters_appended_to_params(self, mock_get, tmp_path, monkeypatch):
        monkeypatch.setattr("bcycle_jp.adapters.estat._CACHE_DIR", tmp_path)
        mock_get.return_value = _make_data_response(_SAMPLE_VALUES)

        src = {
            "stats_data_id": "0003191203",
            "filters": {"cat01": "0010", "area": "00000"},
        }
        self._adapter().fetch(src, start=date(1985, 1, 1))

        call_params = mock_get.call_args.kwargs.get("params", {})
        assert call_params.get("cdCat01") == "0010"
        assert call_params.get("cdArea") == "00000"

    def test_unavailable_without_key(self):
        adapter = EStatAdapter(app_id=None)
        adapter.app_id = None  # 念のため
        with pytest.raises(RuntimeError, match="ESTAT_APP_ID"):
            adapter.fetch({"stats_data_id": "0003191203"}, start=date(1985, 1, 1))

    @patch("bcycle_jp.adapters.estat.requests.get")
    def test_all_null_values_raises(self, mock_get, tmp_path, monkeypatch):
        monkeypatch.setattr("bcycle_jp.adapters.estat._CACHE_DIR", tmp_path)
        null_values = [
            {"@time": "198501", "$": "-"},
            {"@time": "198502", "$": "***"},
            {"@time": "198503", "$": ""},
        ]
        mock_get.return_value = _make_data_response(null_values)
        with pytest.raises(ValueError, match="有効なデータ"):
            self._adapter().fetch(
                {"stats_data_id": "0003191203"}, start=date(1985, 1, 1)
            )


# ---------------------------------------------------------------------------
# EStatAdapter.fetch with time_from_cat (METI IIP format)
# ---------------------------------------------------------------------------


def _make_meta_response(time_cat: str, cat_map: dict[str, str]) -> MagicMock:
    """メタデータレスポンスのモック(CLASS_INF を含む)。"""
    classes = [{"@code": code, "@name": name} for code, name in cat_map.items()]
    mock = MagicMock()
    mock.raise_for_status.return_value = None
    mock.json.return_value = {
        "GET_STATS_DATA": {
            "STATISTICAL_DATA": {
                "RESULT_INF": {"TOTAL_NUMBER": 1},
                "CLASS_INF": {
                    "CLASS_OBJ": [
                        {"@id": time_cat, "CLASS": classes},
                    ]
                },
                "DATA_INF": {"VALUE": []},
            }
        }
    }
    return mock


_SAMPLE_CAT_MAP = {
    "0500100": "201301",
    "0500200": "201302",
    "0500300": "201303",
    "0500400": "201304",
    "9990000": "付加生産ウエイト",  # 非日付エントリ — スキップ対象
}

_SAMPLE_CAT_VALUES = [
    {"@cat01": "0500100", "$": "92.5"},
    {"@cat01": "0500200", "$": "93.0"},
    {"@cat01": "0500300", "$": "-"},    # NULL — 除外
    {"@cat01": "0500400", "$": "94.1"},
    {"@cat01": "9990000", "$": "100.0"},  # 非日付 — スキップ
]


class TestFetchTimeFromCat:
    def _adapter(self):
        return EStatAdapter(app_id="dummy_key")

    @patch("bcycle_jp.adapters.estat.requests.get")
    def test_time_from_cat_returns_series(self, mock_get, tmp_path, monkeypatch):
        monkeypatch.setattr("bcycle_jp.adapters.estat._CACHE_DIR", tmp_path)
        meta = _make_meta_response("cat01", _SAMPLE_CAT_MAP)
        data = _make_data_response(_SAMPLE_CAT_VALUES)
        mock_get.side_effect = [meta, data]

        src = {"stats_data_id": "0004017874", "time_from_cat": "cat01"}
        result = self._adapter().fetch(src, start=date(2013, 1, 1))

        assert isinstance(result, pd.Series)
        assert len(result) == 3  # NULL と非日付を除いた3件
        assert result.index[0] == pd.Timestamp("2013-01-01")
        assert result.iloc[0] == pytest.approx(92.5)
        assert result.iloc[-1] == pytest.approx(94.1)

    @patch("bcycle_jp.adapters.estat.requests.get")
    def test_time_from_cat_skips_non_date_entries(self, mock_get, tmp_path, monkeypatch):
        monkeypatch.setattr("bcycle_jp.adapters.estat._CACHE_DIR", tmp_path)
        meta = _make_meta_response("cat01", _SAMPLE_CAT_MAP)
        data = _make_data_response(_SAMPLE_CAT_VALUES)
        mock_get.side_effect = [meta, data]

        src = {"stats_data_id": "0004017874", "time_from_cat": "cat01"}
        result = self._adapter().fetch(src, start=date(2013, 1, 1))

        # ウエイト行(9990000 → "付加生産ウエイト")が含まれないことを確認
        assert pd.Timestamp("2013-04-01") == result.index[-1]

    @patch("bcycle_jp.adapters.estat.requests.get")
    def test_time_from_cat_uses_cache(self, mock_get, tmp_path, monkeypatch):
        monkeypatch.setattr("bcycle_jp.adapters.estat._CACHE_DIR", tmp_path)
        meta = _make_meta_response("cat01", _SAMPLE_CAT_MAP)
        data = _make_data_response(_SAMPLE_CAT_VALUES)
        mock_get.side_effect = [meta, data]

        src = {"stats_data_id": "0004017874", "time_from_cat": "cat01"}
        adapter = self._adapter()
        adapter.fetch(src, start=date(2013, 1, 1))
        assert mock_get.call_count == 2  # meta + data

        # 2回目はキャッシュから
        mock_get.side_effect = None
        mock_get.return_value = MagicMock()
        adapter.fetch(src, start=date(2013, 1, 1))
        assert mock_get.call_count == 2  # 増えない

    @patch("bcycle_jp.adapters.estat.requests.get")
    def test_time_from_cat_all_null_raises(self, mock_get, tmp_path, monkeypatch):
        monkeypatch.setattr("bcycle_jp.adapters.estat._CACHE_DIR", tmp_path)
        meta = _make_meta_response("cat01", {"0500100": "201301"})
        null_data = _make_data_response([{"@cat01": "0500100", "$": "-"}])
        mock_get.side_effect = [meta, null_data]

        src = {"stats_data_id": "0004017874", "time_from_cat": "cat01"}
        with pytest.raises(ValueError, match="有効なデータ"):
            self._adapter().fetch(src, start=date(2013, 1, 1))
