"""
BOK ECOS (Economic Statistics System) API アダプタ。

実装内容:
  1. ECOS_API_KEY を .env / 環境変数から読み込み
  2. StatisticSearch エンドポイントで月次時系列を取得
  3. item_code1–4 による系列絞り込みに対応
  4. data/ 配下に parquet キャッシュ (再取得回避)
  5. start/end で日付フィルタを適用

参考:
  ECOS Open API 2.0
  https://ecos.bok.or.kr/api/#/
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import date
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

from .base import BaseAdapter

load_dotenv()

_CACHE_DIR = Path(__file__).resolve().parents[3] / "data"
_CACHE_DIR.mkdir(exist_ok=True)

_BASE_URL = "https://ecos.bok.or.kr/api/StatisticSearch"
_NULL_VALUES = frozenset({"", "-", "N/A", ".", "...", " "})


class EcosAdapter(BaseAdapter):
    """BOK ECOS REST API アダプタ。

    source_config フィールド:
      stat_code   str   必須。ECOS 통계표 코드 (例: "901Y009")
      item_code1  str   任意。상위 항목 코드
      item_code2  str   任意。하위 항목 코드
      item_code3  str   任意。
      item_code4  str   任意。
      cycle_type  str   任意。"M" (月次, デフォルト) / "Q" / "A"
    """

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("ECOS_API_KEY")

    def is_available(self) -> bool:
        return bool(self.api_key)

    def fetch(
        self,
        source_config: dict,
        start: date,
        end: date | None = None,
    ) -> pd.Series:
        if not self.is_available():
            raise RuntimeError("ECOS_API_KEY が未設定 (.env を確認)")

        stat_code  = source_config.get("stat_code")
        if not stat_code:
            raise ValueError("source_config に stat_code が必要")

        item_code1 = source_config.get("item_code1", "")
        item_code2 = source_config.get("item_code2", "")
        item_code3 = source_config.get("item_code3", "")
        item_code4 = source_config.get("item_code4", "")
        cycle_type = source_config.get("cycle_type", "M")

        cache_path = self._cache_path(stat_code, item_code1, item_code2,
                                       item_code3, item_code4)

        if cache_path.exists():
            series = pd.read_parquet(cache_path).squeeze()
            series.index = pd.DatetimeIndex(series.index)
            return self._filter_dates(series, start, end)

        series = self._fetch_all(stat_code, cycle_type,
                                  item_code1, item_code2, item_code3, item_code4)
        series.to_frame("value").to_parquet(cache_path)
        return self._filter_dates(series, start, end)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fetch_all(
        self,
        stat_code: str,
        cycle_type: str,
        item_code1: str,
        item_code2: str,
        item_code3: str,
        item_code4: str,
    ) -> pd.Series:
        """全期間を一括取得して月初 DatetimeIndex の pd.Series で返す。

        ECOS API は 1万件/リクエストまで返すため、月次データなら
        800年分以上取得可能。実質的に上限到達なし。
        """
        # 全期間: 1900-01 ～ 現在の翌月まで指定してAPIに任せる
        start_str = "190001"
        end_str   = (pd.Timestamp.now() + pd.DateOffset(months=1)).strftime("%Y%m")

        url = self._build_url(stat_code, cycle_type, start_str, end_str,
                               item_code1, item_code2, item_code3, item_code4)

        resp = requests.get(url, timeout=60)
        resp.raise_for_status()

        body = resp.json()

        # エラーレスポンス判定: {"RESULT": {"CODE": "...", "MESSAGE": "..."}}
        if "RESULT" in body:
            code = body["RESULT"].get("CODE", "")
            msg  = body["RESULT"].get("MESSAGE", "")
            raise ValueError(f"ECOS API エラー [{code}]: {msg}")

        rows = body.get("StatisticSearch", {}).get("row", [])
        if not rows:
            raise ValueError(
                f"stat_code={stat_code!r}: 有効な行が取得できませんでした"
            )

        records: dict[pd.Timestamp, float] = {}
        for row in rows:
            time_str = row.get("TIME", "")
            val_str  = row.get("DATA_VALUE", "")
            if val_str in _NULL_VALUES:
                continue
            try:
                ts = self._parse_time(time_str, cycle_type)
                records[ts] = float(val_str)
            except (ValueError, TypeError):
                continue

        if not records:
            raise ValueError(
                f"stat_code={stat_code!r}: パース可能なデータが0件"
            )

        series = pd.Series(records, dtype=float).sort_index()
        series.index = pd.DatetimeIndex(series.index)
        return series

    def _build_url(
        self,
        stat_code: str,
        cycle_type: str,
        start_str: str,
        end_str: str,
        item_code1: str,
        item_code2: str,
        item_code3: str,
        item_code4: str,
    ) -> str:
        # 末尾の item_code セグメントは空でも / で区切る必要がある
        # 空の場合は省略可だが、途中が空の場合は . で埋める
        items = [item_code1, item_code2, item_code3, item_code4]
        # 後ろの空要素を除去
        while items and items[-1] == "":
            items.pop()
        item_path = "/".join(items) if items else ""

        base = (
            f"{_BASE_URL}/{self.api_key}/json/kr"
            f"/1/100000"
            f"/{stat_code}/{cycle_type}/{start_str}/{end_str}"
        )
        return f"{base}/{item_path}" if item_path else base

    @staticmethod
    def _parse_time(time_str: str, cycle_type: str) -> pd.Timestamp:
        """ECOS の TIME フィールドを月初 Timestamp に変換。

        月次 (M): "202301"     → 2023-01-01
        四半期(Q): "2023Q1"    → 2023-01-01
        年次 (A): "2023"       → 2023-01-01  (年次指標は month=1 に丸める)
        """
        s = str(time_str).strip()
        if cycle_type == "M" and len(s) == 6:
            return pd.Timestamp(year=int(s[:4]), month=int(s[4:6]), day=1)
        if cycle_type == "Q" and "Q" in s:
            year, q = s.split("Q")
            month = (int(q) - 1) * 3 + 1
            return pd.Timestamp(year=int(year), month=month, day=1)
        if cycle_type == "A" and len(s) == 4:
            return pd.Timestamp(year=int(s), month=1, day=1)
        # フォールバック: pandas に任せる
        return pd.Timestamp(s[:6]) if len(s) >= 6 else pd.Timestamp(s)

    @staticmethod
    def _cache_path(
        stat_code: str,
        item_code1: str,
        item_code2: str,
        item_code3: str,
        item_code4: str,
    ) -> Path:
        key_str = json.dumps(
            [stat_code, item_code1, item_code2, item_code3, item_code4],
            sort_keys=True,
        )
        fhash = hashlib.md5(key_str.encode()).hexdigest()[:8]
        return _CACHE_DIR / f"ecos_{stat_code}_{fhash}.parquet"

    @staticmethod
    def _filter_dates(series: pd.Series, start: date, end: date | None) -> pd.Series:
        lo = pd.Timestamp(start)
        hi = pd.Timestamp(end) if end else None
        return series.loc[lo:hi] if hi else series.loc[lo:]
