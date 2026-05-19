"""
e-Stat API アダプタ。

実装内容:
  1. ESTAT_APP_ID を .env / 環境変数から読み込み
  2. stats_data_id == "TBD" なら stats_code + table_name_contains で getStatsList から動的解決
  3. getStatsData で時系列を取得・パース
  4. data/ 配下に parquet キャッシュ (再取得回避)
  5. start/end で日付フィルタを適用

参考:
  API リファレンス v3.0
  https://www.e-stat.go.jp/api/api-info/e-stat-manual3-0
"""
from __future__ import annotations

import hashlib
import json
import os
import warnings
from datetime import date
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

from .base import BaseAdapter

load_dotenv()

_CACHE_DIR = Path(__file__).resolve().parents[3] / "data"
_CACHE_DIR.mkdir(exist_ok=True)

_NULL_VALUES = frozenset({"", "-", "***", "…", "－", "X", "x"})


def _parse_time_code(code: str) -> pd.Timestamp:
    """e-Stat の @time コードを月初の pd.Timestamp に変換する。

    対応形式:
      "202401"     6桁  YYYYMM          月次の主流形式
      "20240101"   8桁  YYYYMMDD        一部統計(月初に丸める)
      "2024000101" 10桁 YYYY + "00" + MM + "01"  旧形式・一部調査
    """
    s = str(code).strip()
    n = len(s)
    if n == 6:
        return pd.Timestamp(year=int(s[:4]), month=int(s[4:6]), day=1)
    elif n == 8:
        return pd.Timestamp(year=int(s[:4]), month=int(s[4:6]), day=1)
    elif n == 10:
        year = int(s[:4])
        month = int(s[6:8])
        if month == 0:
            raise ValueError(f"年次データの時間コードは対象外: {code!r}")
        return pd.Timestamp(year=year, month=month, day=1)
    else:
        raise ValueError(f"不明な時間コード形式: {code!r} (len={n})")


class EStatAdapter(BaseAdapter):
    """e-Stat (政府統計の総合窓口) REST API v3.0 アダプタ。"""

    BASE_URL = "https://api.e-stat.go.jp/rest/3.0/app/json"

    def __init__(self, app_id: str | None = None):
        self.app_id = app_id or os.environ.get("ESTAT_APP_ID")

    def is_available(self) -> bool:
        return bool(self.app_id)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def resolve_stats_data_id(
        self,
        stats_code: str,
        table_name_contains: list[str] | str | None = None,
    ) -> str:
        """stats_code から getStatsList を叩いて stats_data_id を返す。

        Parameters
        ----------
        stats_code : str
            e-Stat の統計コード (yaml の stats_code フィールド)
        table_name_contains : list[str] | str | None
            タイトルに含まれるべきキーワード。全て含む最初の表を選ぶ。
            None なら最新(先頭)の表を返す。

        Returns
        -------
        str
            stats_data_id (getStatsList の TABLE_INF[@id])
        """
        params = {
            "appId": self.app_id,
            "statsCode": stats_code,
            "limit": 100,
        }
        resp = requests.get(
            f"{self.BASE_URL}/getStatsList", params=params, timeout=30
        )
        resp.raise_for_status()

        list_inf = resp.json()["GET_STATS_LIST"]["DATALIST_INF"]
        tables = list_inf.get("TABLE_INF", [])
        if isinstance(tables, dict):
            tables = [tables]
        if not tables:
            raise ValueError(
                f"stats_code={stats_code!r} に対応する表が見つかりません"
            )

        if table_name_contains:
            keywords = (
                [table_name_contains]
                if isinstance(table_name_contains, str)
                else list(table_name_contains)
            )
            for tbl in tables:
                title = tbl.get("TITLE", "")
                if isinstance(title, dict):
                    title = title.get("$", "")
                if all(kw in title for kw in keywords):
                    return str(tbl["@id"])
            warnings.warn(
                f"stats_code={stats_code!r}: キーワード {keywords} に一致する表なし。"
                f"先頭の表 {tables[0]['@id']!r} を使用。",
                stacklevel=2,
            )

        return str(tables[0]["@id"])

    def fetch(
        self,
        source_config: dict,
        start: date,
        end: date | None = None,
    ) -> pd.Series:
        """指標を取得して DatetimeIndex の pd.Series で返す。

        source_config の必須/任意フィールド:
          stats_code            str   統計コード
          stats_data_id         str   "TBD" なら getStatsList で動的解決
          table_name_contains   list  テーブル選択キーワード(任意)
          filters               dict  カテゴリフィルタ {cat02: "1001000", ...}(任意)
          time_from_cat         str   日付がカテゴリのname属性に入っている場合のカテゴリID
                                      例: "cat01"  →  @cat01 の CLASS name が YYYYMM
        """
        if not self.is_available():
            raise RuntimeError("ESTAT_APP_ID が未設定 (.env を確認)")

        stats_data_id = source_config.get("stats_data_id", "TBD")
        stats_code = source_config.get("stats_code", "")

        if not stats_data_id or stats_data_id == "TBD":
            keywords = source_config.get("table_name_contains")
            stats_data_id = self.resolve_stats_data_id(stats_code, keywords)

        filters: dict = source_config.get("filters", {}) or {}
        time_from_cat: str | None = source_config.get("time_from_cat")

        cache_key = {**filters, "__time_from_cat__": time_from_cat or ""}
        cache_path = self._cache_path(stats_data_id, cache_key)

        if cache_path.exists():
            series = pd.read_parquet(cache_path).squeeze()
            series.index = pd.DatetimeIndex(series.index)
            return self._filter_dates(series, start, end)

        if time_from_cat:
            series = self._fetch_series_time_from_cat(
                stats_data_id, filters, time_from_cat
            )
        else:
            series = self._fetch_series(stats_data_id, filters)

        series.to_frame("value").to_parquet(cache_path)
        return self._filter_dates(series, start, end)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fetch_series(self, stats_data_id: str, filters: dict) -> pd.Series:
        """標準形式: @time フィールドに日付が入るテーブル用。"""
        params: dict[str, str | int] = {
            "appId": self.app_id,
            "statsDataId": stats_data_id,
            "metaGetFlg": "N",
            "cntGetFlg": "N",
        }
        # filters: {cat01: "0010", area: "00000"} → cdCat01=0010, cdArea=00000
        for k, v in filters.items():
            key = "cd" + k[0].upper() + k[1:]
            params[key] = v

        resp = requests.get(
            f"{self.BASE_URL}/getStatsData", params=params, timeout=60
        )
        resp.raise_for_status()

        stat_data = resp.json()["GET_STATS_DATA"]["STATISTICAL_DATA"]
        values = stat_data["DATA_INF"]["VALUE"]
        if isinstance(values, dict):
            values = [values]

        records: dict[pd.Timestamp, float] = {}
        for v in values:
            raw = v.get("$", "")
            if raw in _NULL_VALUES:
                continue
            try:
                ts = _parse_time_code(v["@time"])
                records[ts] = float(raw)
            except (ValueError, KeyError, TypeError):
                continue

        if not records:
            raise ValueError(
                f"stats_data_id={stats_data_id!r}: 有効なデータが取得できませんでした"
            )

        series = pd.Series(records, dtype=float).sort_index()
        series.index = pd.DatetimeIndex(series.index)
        return series

    def _fetch_series_time_from_cat(
        self, stats_data_id: str, filters: dict, time_cat: str
    ) -> pd.Series:
        """time_from_cat 形式: @time が空で日付がカテゴリの name 属性に格納されるテーブル用。

        経産省鉱工業指数など一部のMETI統計で使われる形式。
        cat01 の CLASS name が '201301' などの YYYYMM 文字列。
        """
        # 1. メタデータ取得 → cat_code → YYYYMM マップを構築
        meta_resp = requests.get(
            f"{self.BASE_URL}/getStatsData",
            params={
                "appId": self.app_id,
                "statsDataId": stats_data_id,
                "metaGetFlg": "Y",
                "cntGetFlg": "Y",
                "limit": 1,
            },
            timeout=30,
        )
        meta_resp.raise_for_status()
        class_objs = (
            meta_resp.json()["GET_STATS_DATA"]["STATISTICAL_DATA"]
            .get("CLASS_INF", {})
            .get("CLASS_OBJ", [])
        )
        if isinstance(class_objs, dict):
            class_objs = [class_objs]

        cat_map: dict[str, str] = {}
        for c in class_objs:
            if c.get("@id") == time_cat:
                classes = c.get("CLASS", [])
                if isinstance(classes, dict):
                    classes = [classes]
                for item in classes:
                    cat_map[item.get("@code", "")] = item.get("@name", "")
                break

        # 2. データ取得
        data_params: dict[str, str | int] = {
            "appId": self.app_id,
            "statsDataId": stats_data_id,
            "metaGetFlg": "N",
            "cntGetFlg": "N",
        }
        for k, v in filters.items():
            key = "cd" + k[0].upper() + k[1:]
            data_params[key] = v

        data_resp = requests.get(
            f"{self.BASE_URL}/getStatsData", params=data_params, timeout=60
        )
        data_resp.raise_for_status()

        values = (
            data_resp.json()["GET_STATS_DATA"]["STATISTICAL_DATA"]
            ["DATA_INF"]["VALUE"]
        )
        if isinstance(values, dict):
            values = [values]

        # 3. cat_code → YYYYMM → Timestamp に変換
        records: dict[pd.Timestamp, float] = {}
        for v in values:
            raw = v.get("$", "")
            if raw in _NULL_VALUES:
                continue
            cat_code = v.get(f"@{time_cat}", "")
            date_str = cat_map.get(cat_code, "")
            if len(date_str) != 6 or not date_str.isdigit():
                continue  # ウエイト行など非日付エントリをスキップ
            try:
                ts = _parse_time_code(date_str)
                records[ts] = float(raw)
            except (ValueError, TypeError):
                continue

        if not records:
            raise ValueError(
                f"stats_data_id={stats_data_id!r}: time_from_cat={time_cat!r}: "
                "有効なデータが取得できませんでした"
            )

        series = pd.Series(records, dtype=float).sort_index()
        series.index = pd.DatetimeIndex(series.index)
        return series

    @staticmethod
    def _cache_path(stats_data_id: str, filters: dict) -> Path:
        if filters:
            fhash = hashlib.md5(
                json.dumps(filters, sort_keys=True).encode()
            ).hexdigest()[:8]
            fname = f"estat_{stats_data_id}_{fhash}.parquet"
        else:
            fname = f"estat_{stats_data_id}.parquet"
        return _CACHE_DIR / fname

    @staticmethod
    def _filter_dates(
        series: pd.Series, start: date, end: date | None
    ) -> pd.Series:
        lo = pd.Timestamp(start)
        hi = pd.Timestamp(end) if end else None
        return series.loc[lo:hi] if hi else series.loc[lo:]
