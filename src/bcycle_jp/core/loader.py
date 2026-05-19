"""
yaml の指標定義を読み、適切なアダプタで取得して整形するエントリ。
"""
from __future__ import annotations

import os
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from bcycle_jp.adapters.base import BaseAdapter
from bcycle_jp.adapters.registry import get_adapter
from bcycle_jp.core.normalize import apply_transform


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_indicator(
    indicator_def: dict,
    prefer: str = "estat",
    start: date = date(1985, 1, 1),
    end: date | None = None,
) -> pd.Series:
    """yaml の1指標エントリから時系列を取得 + transform を適用。

    `prefer` ソースが使えない/未設定なら sources の他候補にフォールバック。
    """
    indicator_id = indicator_def["id"]
    transform = indicator_def.get("transform", "level")
    sources = indicator_def.get("sources", {})

    # yaml の prefer フィールドで指標単位のソースを上書きできる
    prefer = indicator_def.get("prefer", prefer)

    # splice モード: 複数ソースを日付区間で結合
    if prefer == "splice":
        from .composite_sources import fetch_splice
        raw = fetch_splice(sources, global_start=start, global_end=end)
        raw.name = indicator_id
        return apply_transform(raw, transform)

    # フォールバック順: prefer → 残りのキー順
    source_order = [prefer] + [k for k in sources.keys() if k != prefer]

    last_err: Exception | None = None
    for src in source_order:
        if src not in sources:
            continue
        try:
            adapter: BaseAdapter = get_adapter(src)
            if not adapter.is_available():
                continue
            raw = adapter.fetch(sources[src], start=start, end=end)
            raw.name = indicator_id
            return apply_transform(raw, transform)
        except NotImplementedError:
            # アダプタ未実装はスキップ
            continue
        except Exception as e:
            last_err = e
            continue

    raise RuntimeError(
        f"指標 '{indicator_id}' をどのソースからも取得できません。"
        f"最後のエラー: {last_err}"
    )


def get_all_indicators(
    indicators_yaml_path: str | Path,
    prefer: str | None = None,
    start: date = date(1985, 1, 1),
    end: date | None = None,
) -> dict[str, pd.Series]:
    """yaml 全指標を取得して辞書で返す。derived は別途処理。"""
    prefer = prefer or os.environ.get("DATA_SOURCE_PREFER", "estat")
    cfg = load_yaml(indicators_yaml_path)

    out: dict[str, pd.Series] = {}
    for ind in cfg["indicators"]:
        if ind.get("transform") == "derived":
            # 派生指標は他指標を使うので後段で計算
            continue
        try:
            out[ind["id"]] = get_indicator(ind, prefer=prefer, start=start, end=end)
        except RuntimeError as e:
            print(f"[WARN] {ind['id']}: {e}")
    return out


def compute_derived_indicators(
    base: dict[str, pd.Series],
    indicators_yaml_path: str | Path,
) -> dict[str, pd.Series]:
    """derived 指標を base 辞書から計算して返す。

    formula は "A - B" または "A + B" の二項演算のみサポート。
    一方のコンポーネントが base に存在しない場合は WARN してスキップ。
    """
    cfg = load_yaml(indicators_yaml_path)
    out: dict[str, pd.Series] = {}

    for ind in cfg["indicators"]:
        if ind.get("transform") != "derived":
            continue
        ind_id = ind["id"]
        formula = ind.get("formula", "")

        # "A - B" / "A + B" をパース
        op, a_name, b_name = None, None, None
        if " - " in formula:
            parts = formula.split(" - ", 1)
            op, a_name, b_name = "-", parts[0].strip(), parts[1].strip()
        elif " + " in formula:
            parts = formula.split(" + ", 1)
            op, a_name, b_name = "+", parts[0].strip(), parts[1].strip()

        if op is None:
            print(f"[WARN] {ind_id}: サポート外の formula '{formula}'")
            continue

        missing = [n for n in [a_name, b_name] if n not in base]
        if missing:
            print(f"[WARN] {ind_id}: formula コンポーネント未取得 {missing}")
            continue

        a, b = base[a_name], base[b_name]
        result = (a - b) if op == "-" else (a + b)
        result = result.dropna()
        result.name = ind_id
        out[ind_id] = result

    return out
