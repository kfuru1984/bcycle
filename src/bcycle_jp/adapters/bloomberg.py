"""
Bloomberg アダプタ。

⚠️ Claude Code への引き渡しポイント

実装は blpapi(BBComm 経由)または BQL(bquant 経由)のいずれか。
Terminal が手元にある端末でしか動かないので、
is_available() は import 時のエラーをハンドリングして False を返す。

参考:
  https://github.com/msitt/blpapi-python
  https://bquant.bloomberg.com/
"""
from __future__ import annotations

from datetime import date

import pandas as pd

from .base import BaseAdapter


class BloombergAdapter(BaseAdapter):
    def __init__(self):
        try:
            import blpapi  # noqa: F401
            self._available = True
        except ImportError:
            self._available = False

    def is_available(self) -> bool:
        return self._available

    def fetch(
        self,
        source_config: dict,
        start: date,
        end: date | None = None,
    ) -> pd.Series:
        if not self._available:
            raise RuntimeError(
                "blpapi が利用不可。Bloomberg Terminal 環境で実行してください"
            )

        ticker = source_config.get("ticker")
        if not ticker:
            raise ValueError("source_config に ticker が必要")

        # TODO[Claude Code]:
        #   blpapi の HistoricalDataRequest で px_last (or
        #   indicator value) を取得して pd.Series で返す
        raise NotImplementedError("Claude Code で実装してください")
