"""
データソース・アダプタの基底クラス。

全てのデータソース(e-Stat / FRED / Bloomberg など)は
このインターフェースを実装する。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date

import pandas as pd


class BaseAdapter(ABC):
    """全アダプタ共通インターフェース。

    yaml の `sources.<source_name>` ブロックを `source_config` として受け取り、
    日付インデックスの pd.Series を返す。
    """

    @abstractmethod
    def fetch(
        self,
        source_config: dict,
        start: date,
        end: date | None = None,
    ) -> pd.Series:
        """指標を取得して時系列で返す。

        Parameters
        ----------
        source_config : dict
            yaml の sources.<source_name> サブツリー。
            例: {"stats_code": "00550010", "stats_data_id": "TBD"}
        start : date
            取得開始日
        end : date | None
            取得終了日(None なら最新まで)

        Returns
        -------
        pd.Series
            DatetimeIndex で値を持つ系列。name は indicator id を入れる。
        """
        raise NotImplementedError

    def is_available(self) -> bool:
        """このアダプタが利用可能か(APIキー設定済み等)。

        サブクラスでオーバーライドして、認証情報の有無をチェックする。
        """
        return True
