"""
OECD SDMX-JSON API アダプタ。

認証不要。SDMX 2.0 JSON フォーマットでデータを取得する。

参考:
  OECD SDMX REST API
  https://sdmx.oecd.org/public/rest/

source_config フィールド:
  dataflow  str   必須。SDMX dataflow ID (例: "OECD.SDD.TPS,DSD_PRICES@DF_PRICES_N_TXCP01_NRG,1.0")
  key       str   必須。SDMX キー (例: "KOR.M.N.CPI.PA._TXCP01_NRG.N.GY")
                  ワイルドカードに "." を使用可 (例: "KOR.........")
  select    dict  任意。複数系列が返った場合に絞り込む次元フィルタ
                  例: {"MEASURE": "UNE_LF_M", "ADJUSTMENT": "Y", "SEX": "_T"}
                  一致する系列が複数残った場合は最初の1本を使用。
"""
from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path

import pandas as pd
import requests

from .base import BaseAdapter

_CACHE_DIR = Path(__file__).resolve().parents[3] / "data"
_CACHE_DIR.mkdir(exist_ok=True)

_BASE_URL = "https://sdmx.oecd.org/public/rest/data"
_HEADERS  = {"Accept": "application/vnd.sdmx.data+json;version=2.0"}


class OecdAdapter(BaseAdapter):
    """OECD SDMX-JSON REST API アダプタ。認証不要。"""

    def is_available(self) -> bool:
        return True

    def fetch(
        self,
        source_config: dict,
        start: date,
        end: date | None = None,
    ) -> pd.Series:
        dataflow = source_config.get("dataflow")
        key      = source_config.get("key", "")
        select   = source_config.get("select", {}) or {}

        if not dataflow:
            raise ValueError("source_config に dataflow が必要")

        cache_path = self._cache_path(dataflow, key, select)

        if cache_path.exists():
            series = pd.read_parquet(cache_path).squeeze()
            series.index = pd.DatetimeIndex(series.index)
            return self._filter_dates(series, start, end)

        series = self._fetch_all(dataflow, key, select)
        series.to_frame("value").to_parquet(cache_path)
        return self._filter_dates(series, start, end)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fetch_all(
        self,
        dataflow: str,
        key: str,
        select: dict[str, str],
    ) -> pd.Series:
        params: dict[str, str] = {"dimensionAtObservation": "TIME_PERIOD"}

        url = f"{_BASE_URL}/{dataflow}/{key}"
        resp = requests.get(url, headers=_HEADERS, params=params, timeout=60)
        resp.raise_for_status()

        body = resp.json()
        data_node = body.get("data", {})

        structures = data_node.get("structures", [])
        datasets   = data_node.get("dataSets", [])

        if not structures or not datasets:
            raise ValueError(f"dataflow={dataflow!r}: レスポンスに data.structures / dataSets が存在しません")

        struct = structures[0]
        dims   = struct.get("dimensions", {})

        # 系列次元のインデックス → コード マップを構築
        series_dims = dims.get("series", [])
        dim_value_maps: list[list[str]] = []
        for d in series_dims:
            dim_value_maps.append([v["id"] for v in d.get("values", [])])

        # 観測時間インデックス → 期間文字列 マップ
        obs_dims = dims.get("observation", [])
        time_dim = next(
            (d for d in obs_dims if d.get("id") == "TIME_PERIOD"),
            None,
        )
        if time_dim is None:
            raise ValueError(f"dataflow={dataflow!r}: TIME_PERIOD 次元が見つかりません")
        time_map = {str(i): v["id"] for i, v in enumerate(time_dim.get("values", []))}

        # select フィルタ用: 次元名 → 位置インデックス
        dim_name_to_pos = {d["id"]: i for i, d in enumerate(series_dims)}

        dataset = datasets[0]
        all_series = dataset.get("series", {})

        # 複数系列から select で絞り込む
        matched: dict[str, dict] = {}
        for series_key, series_data in all_series.items():
            idx_parts = series_key.split(":")
            # select フィルタ適用
            match = True
            for dim_name, required_code in select.items():
                pos = dim_name_to_pos.get(dim_name)
                if pos is None or pos >= len(idx_parts):
                    match = False
                    break
                actual_code = dim_value_maps[pos][int(idx_parts[pos])]
                if actual_code != required_code:
                    match = False
                    break
            if match:
                matched[series_key] = series_data

        if not matched:
            raise ValueError(
                f"dataflow={dataflow!r} key={key!r}: select={select} に一致する系列なし"
            )
        if len(matched) > 1 and select:
            # 絞り込み後もまだ複数 → 警告して最初を使用
            import warnings
            warnings.warn(
                f"dataflow={dataflow!r}: select={select} で {len(matched)} 系列マッチ。"
                f"最初の系列を使用: {list(matched.keys())[0]}",
                stacklevel=3,
            )

        # 最初にマッチした系列の観測値を抽出
        chosen_data = next(iter(matched.values()))
        records: dict[pd.Timestamp, float] = {}
        for obs_idx, obs_vals in chosen_data.get("observations", {}).items():
            time_str = time_map.get(obs_idx)
            if not time_str:
                continue
            value = obs_vals[0]
            if value is None:
                continue
            try:
                ts = self._parse_time(time_str)
                records[ts] = float(value)
            except (ValueError, TypeError):
                continue

        if not records:
            raise ValueError(
                f"dataflow={dataflow!r} key={key!r}: 有効な観測値が0件"
            )

        series = pd.Series(records, dtype=float).sort_index()
        series.index = pd.DatetimeIndex(series.index)
        return series

    @staticmethod
    def _parse_time(time_str: str) -> pd.Timestamp:
        """SDMX 期間文字列 (2023-01 / 2023-Q1 / 2023) → 月初 Timestamp。"""
        s = str(time_str).strip()
        if len(s) == 7 and s[4] == "-" and s[5] != "Q":
            # YYYY-MM
            return pd.Timestamp(year=int(s[:4]), month=int(s[5:7]), day=1)
        if "Q" in s:
            # YYYY-Q1 / YYYYQ1
            s = s.replace("-", "")
            year, q = int(s[:4]), int(s[5])
            return pd.Timestamp(year=year, month=(q - 1) * 3 + 1, day=1)
        if len(s) == 4:
            return pd.Timestamp(year=int(s), month=1, day=1)
        # fallback: pandas に委ねる
        return pd.Timestamp(s)

    @staticmethod
    def _cache_path(dataflow: str, key: str, select: dict[str, str]) -> Path:
        select_str = json.dumps(select, sort_keys=True) if select else ""
        raw = f"{dataflow}|{key}|{select_str}"
        fhash = hashlib.md5(raw.encode()).hexdigest()[:10]
        safe = dataflow.replace(",", "_").replace("@", "_").replace(".", "_")[:40]
        return _CACHE_DIR / f"oecd_{safe}_{fhash}.parquet"

    @staticmethod
    def _filter_dates(series: pd.Series, start: date, end: date | None) -> pd.Series:
        lo = pd.Timestamp(start)
        hi = pd.Timestamp(end) if end else None
        return series.loc[lo:hi] if hi else series.loc[lo:]
