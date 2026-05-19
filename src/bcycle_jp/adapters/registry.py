"""
ソース名 → アダプタ実装クラスのディスパッチャ。

新しいソースを追加する時はここに登録。
"""
from __future__ import annotations

from .base import BaseAdapter
from .estat import EStatAdapter
from .fred import FredAdapter
from .bloomberg import BloombergAdapter
from .ecos import EcosAdapter
from .oecd import OecdAdapter


ADAPTERS: dict[str, type[BaseAdapter]] = {
    "estat": EStatAdapter,
    "fred": FredAdapter,
    "fred_oecd": FredAdapter,   # FredAdapter のエイリアス。series_id は yaml 側で指定
    "bloomberg": BloombergAdapter,
    "ecos": EcosAdapter,
    "oecd": OecdAdapter,
}


def get_adapter(source: str) -> BaseAdapter:
    """ソース名からアダプタ・インスタンスを生成して返す。"""
    if source not in ADAPTERS:
        raise ValueError(f"未登録のデータソース: {source}")
    return ADAPTERS[source]()
