"""
在庫循環分析スクリプト。

実行例:
    python scripts/run_inventory_cycle.py
    python scripts/run_inventory_cycle.py --countries jp us
    python scripts/run_inventory_cycle.py --countries jp us eu cn --start-year 2018
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from bcycle_jp.analysis.inventory_cycle import run

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

VALID_COUNTRIES = {"jp", "us", "eu", "cn"}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="在庫循環分析")
    parser.add_argument(
        "--countries", nargs="+",
        default=["jp", "us", "eu", "cn"],
        help="対象国 (jp us eu cn の任意の組み合わせ)",
    )
    parser.add_argument(
        "--start-year", type=int, default=2018,
        help="チャート表示開始年 (デフォルト: 2018)",
    )
    args = parser.parse_args()

    countries = [c.lower() for c in args.countries]
    invalid   = set(countries) - VALID_COUNTRIES
    if invalid:
        parser.error(f"不明な国コード: {invalid}  有効値: {VALID_COUNTRIES}")

    log.info("在庫循環分析 開始: countries=%s start_year=%d", countries, args.start_year)

    results = run(countries=countries, start_year=args.start_year)

    ok  = [c for c, d in results.items() if d.get("ok")]
    err = [c for c, d in results.items() if not d.get("ok")]

    if ok:
        log.info("完了 (成功): %s", ok)
    if err:
        log.warning("失敗: %s", err)
        for c in err:
            log.warning("  %s: %s", c, results[c].get("error", "unknown"))

    sys.exit(1 if err and not ok else 0)
