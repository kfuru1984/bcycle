"""
Ken French Data Library アダプタ。

認証不要。ZIPをダウンロードして CSV をパースし、
data/french/{hash}.parquet にキャッシュする。

参考: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
"""
from __future__ import annotations

import hashlib
import io
import zipfile
from pathlib import Path

import pandas as pd
import requests

_CACHE_DIR = Path(__file__).resolve().parents[3] / "data" / "french"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)

_BASE_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"


def _cache_path(zip_name: str, section: str | None) -> Path:
    key = f"{zip_name}|{section or ''}"
    h = hashlib.md5(key.encode()).hexdigest()[:10]
    safe = zip_name.replace(".", "_").replace("-", "_")[:40]
    return _CACHE_DIR / f"french_{safe}_{h}.parquet"


def _download_csv(zip_name: str) -> str:
    url = _BASE_URL + zip_name
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    zf = zipfile.ZipFile(io.BytesIO(resp.content))
    return zf.read(zf.namelist()[0]).decode("latin-1")


def _parse_monthly(content: str, section_keyword: str | None = None) -> pd.DataFrame:
    """
    Parse a French Data Library CSV and return monthly (YYYYMM) data.

    section_keyword: if given, seek that string first (e.g. "Value Weighted Returns -- Monthly").
    Monthly rows are identified by a 6-digit date (190001–209912).
    Annual rows (4-digit YYYY) are silently skipped.
    Missing-value sentinels (-99.99, -999) become NaN.
    """
    lines = content.splitlines()

    # ── locate start of section ─────────────────────────────────────────────
    start = 0
    if section_keyword:
        for i, line in enumerate(lines):
            if section_keyword.lower() in line.lower():
                start = i + 1
                break
        else:
            raise ValueError(f"Section '{section_keyword}' not found in CSV")

    # ── find column-header line (starts with comma) ──────────────────────────
    header: list[str] | None = None
    data_start = start
    for i in range(start, min(start + 30, len(lines))):
        stripped = lines[i].strip()
        if stripped.startswith(","):
            header = [c.strip() for c in stripped.split(",")]
            data_start = i + 1
            break

    if header is None:
        raise ValueError("Column header not found")

    # ── parse data rows ──────────────────────────────────────────────────────
    records: list[dict] = []
    in_data = False

    for line in lines[data_start:]:
        stripped = line.strip()

        if not stripped:
            # blank line ends the section when we're inside a named section
            if section_keyword and in_data:
                break
            continue

        parts = [p.strip() for p in line.split(",")]
        if not parts[0]:
            continue

        try:
            date_val = int(parts[0].replace(" ", ""))
        except ValueError:
            if in_data:
                break  # text after data = end
            continue

        # Monthly: 6-digit YYYYMM
        if not (190001 <= date_val <= 209912):
            continue  # skip annual (YYYY) rows

        year, month = date_val // 100, date_val % 100
        if not (1 <= month <= 12):
            continue

        in_data = True
        row: dict = {"date": pd.Timestamp(year=year, month=month, day=1)}
        for j, col in enumerate(header[1:], 1):
            if not col:
                continue
            try:
                val = float(parts[j]) if j < len(parts) and parts[j] else float("nan")
                row[col] = float("nan") if val <= -99.0 else val
            except (ValueError, TypeError):
                row[col] = float("nan")
        records.append(row)

    if not records:
        raise ValueError(f"No monthly data found (section={section_keyword!r})")

    df = pd.DataFrame(records).set_index("date")
    df.index = pd.DatetimeIndex(df.index)
    return df.sort_index()


def fetch(zip_name: str, section_keyword: str | None = None) -> pd.DataFrame:
    """
    Download French ZIP, parse monthly data, cache as Parquet, return DataFrame.

    Parameters
    ----------
    zip_name        : filename in the French /ftp/ directory (e.g. "Japan_5_Factors_CSV.zip")
    section_keyword : substring identifying the desired section (for multi-section CSVs)
    """
    cache = _cache_path(zip_name, section_keyword)
    if cache.exists():
        df = pd.read_parquet(cache)
        df.index = pd.DatetimeIndex(df.index)
        return df

    content = _download_csv(zip_name)
    df = _parse_monthly(content, section_keyword)
    df.to_parquet(cache)
    return df
