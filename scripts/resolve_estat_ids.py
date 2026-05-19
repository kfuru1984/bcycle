"""
e-Stat stats_data_id 一括解決スクリプト。

config/indicators.yaml の stats_data_id: "TBD" を
getStatsList API で動的に解決し、yaml をインプレース更新する。

使い方:
    python scripts/resolve_estat_ids.py           # ドライラン(解決のみ表示)
    python scripts/resolve_estat_ids.py --write   # yaml に書き込み
    python scripts/resolve_estat_ids.py --list-tables <stats_code>  # 表一覧表示
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bcycle_jp.adapters.estat import EStatAdapter  # noqa: E402


def list_tables(stats_code: str, adapter: EStatAdapter) -> None:
    """stats_code に紐づく表一覧を表示する(キーワード選定の参考用)。"""
    import requests

    params = {"appId": adapter.app_id, "statsCode": stats_code, "limit": 100}
    resp = requests.get(f"{adapter.BASE_URL}/getStatsList", params=params, timeout=30)
    resp.raise_for_status()

    tables = resp.json()["GET_STATS_LIST"]["DATALIST_INF"].get("TABLE_INF", [])
    if isinstance(tables, dict):
        tables = [tables]

    print(f"\n=== stats_code: {stats_code} ({len(tables)} 表) ===")
    for t in tables:
        tid = t.get("@id", "")
        title = t.get("TITLE", "")
        if isinstance(title, dict):
            title = title.get("$", "")
        survey = t.get("SURVEY_DATE", "")
        print(f"  {tid}  {title}  [{survey}]")


def update_stats_data_id(yaml_text: str, stats_code: str, new_id: str) -> str:
    """yaml テキスト中の特定 stats_code 下の stats_data_id: "TBD" を置換する。

    コメントや構造を破壊しない外科的なテキスト置換を使用する。
    """
    pattern = re.compile(
        r'(stats_code:\s*"'
        + re.escape(stats_code)
        + r'"[^\n]*\n(?:\s+[^\n]*\n)*?\s*stats_data_id:\s*)"TBD"',
        re.MULTILINE,
    )
    replaced, count = pattern.subn(r'\g<1>"' + new_id + '"', yaml_text)
    if count == 0:
        # Fallback: 直前に stats_code がある場合
        pattern2 = re.compile(
            r'(stats_code:\s*"'
            + re.escape(stats_code)
            + r'"[^\n]*\n\s*stats_data_id:\s*)"TBD"',
            re.MULTILINE,
        )
        replaced, count = pattern2.subn(r'\g<1>"' + new_id + '"', yaml_text)
    return replaced


def resolve_all(yaml_path: Path, adapter: EStatAdapter, write: bool = False) -> None:
    """yaml の TBD を全て解決し、必要なら yaml を更新する。"""
    with open(yaml_path, "r", encoding="utf-8") as f:
        yaml_text = f.read()

    cfg = yaml.safe_load(yaml_text)
    updated = yaml_text
    any_resolved = False

    for ind in cfg.get("indicators", []):
        estat_src = ind.get("sources", {}).get("estat")
        if not estat_src:
            continue
        sid = estat_src.get("stats_data_id", "")
        if sid and sid != "TBD":
            print(f"[SKIP] {ind['id']}: stats_data_id={sid!r} (既設定)")
            continue

        stats_code = estat_src.get("stats_code", "")
        keywords = estat_src.get("table_name_contains")
        print(f"\n[RESOLVE] {ind['id']} (stats_code={stats_code!r})")
        if keywords:
            print(f"  キーワード: {keywords}")

        try:
            new_id = adapter.resolve_stats_data_id(stats_code, keywords)
            print(f"  → stats_data_id = {new_id!r}")
            if write:
                updated = update_stats_data_id(updated, stats_code, new_id)
                any_resolved = True
        except Exception as e:
            print(f"  [ERROR] {e}")

    if write and any_resolved:
        with open(yaml_path, "w", encoding="utf-8") as f:
            f.write(updated)
        print(f"\nyaml 更新完了: {yaml_path}")
    elif write:
        print("\n更新対象なし。")
    else:
        print("\n--write を指定すると yaml に書き込みます。")


def main() -> None:
    parser = argparse.ArgumentParser(description="e-Stat stats_data_id 一括解決")
    parser.add_argument("--write", action="store_true", help="yaml を更新する")
    parser.add_argument(
        "--list-tables",
        metavar="STATS_CODE",
        help="指定 stats_code の表一覧を表示",
    )
    args = parser.parse_args()

    adapter = EStatAdapter()
    if not adapter.is_available():
        print("ERROR: ESTAT_APP_ID が未設定です。.env を確認してください。")
        sys.exit(1)

    if args.list_tables:
        list_tables(args.list_tables, adapter)
        return

    yaml_path = ROOT / "config" / "indicators.yaml"
    resolve_all(yaml_path, adapter, write=args.write)


if __name__ == "__main__":
    main()
