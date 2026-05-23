# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Commands

```bash
# Install (editable, with dev extras)
uv pip install -e ".[dev]"

# Run tests
pytest tests/

# Run a single test file
pytest tests/test_normalize.py -v

# Full cycle run (JP / US / KR)
python scripts/run_cycle.py --country jp
python scripts/run_cycle.py --country us
python scripts/run_cycle.py --country kr

# Inventory cycle analysis
python scripts/run_inventory_cycle.py --countries jp us

# Regenerate dashboard HTML
python scripts/generate_dashboard.py

# Regenerate monthly report
python scripts/generate_report.py
```

**Required environment variables** (copy `.env.example` → `.env`):
- `ESTAT_APP_ID` — e-Stat API (JP indicators)
- `FRED_API_KEY` — FRED API (US/KR/global proxies)
- `ECOS_API_KEY` — 韓国銀行 BOK (KR indicators)
- `DATA_SOURCE_PREFER` — optional; default `"estat"` for JP, overrides per-indicator `prefer` field

---

## Architecture

### Data pipeline

```
config/{country}/indicators.yaml
        │
        ▼
core/loader.py  ──  get_adapter(src) from adapters/registry.py
        │                   │
        │           estat / fred / oecd / ecos (all return month-start pd.Series)
        │
        ▼
core/normalize.py   apply_transform() → yoy / mom / level / z_score / pct_rank
        │
        ▼
core/composite.py   compute_level()     → 0-100 percentile composite
                    compute_momentum()  → 3M z-score of level
        │
        ▼
core/classify.py    classify_stage()    → 5-stage with hysteresis
                    stage_confidence()  → 0-1 agreement ratio
```

**5 stages:** 回復 / 上昇 / 成熟 / 軟化 / 下降, determined by `level × momentum` quadrant.
Thresholds `level_low=33`, `level_high=60`, `hysteresis=2` are in `config/{country}/settings.yaml`.

### Adapter pattern

All adapters inherit `BaseAdapter` (`adapters/base.py`) with two methods:
- `fetch(source_config: dict, start, end) → pd.Series` — always month-start index
- `is_available() → bool` — checks env var / library availability

`adapters/registry.py::get_adapter(src)` dispatches by string key (`"estat"`, `"fred"`, `"oecd"`, `"ecos"`, `"bloomberg"`).

`loader.py` tries sources in order: `prefer` first → remaining keys → raises `RuntimeError`. Setting `prefer: splice` in YAML triggers date-range splicing across multiple sources via `composite_sources.py`.

**Caching:** each adapter writes `data/{adapter}_{key_hash}.parquet`. Re-runs use cached files; delete them to force a refresh.

### YAML indicator definition

Each entry in `config/{country}/indicators.yaml`:
```yaml
- id: ip_yoy
  transform: yoy          # yoy / mom / level / z_score / pct_rank / derived
  weight: 1.0
  prefer: estat           # or fred / oecd / splice
  sources:
    estat: { stats_data_id: "0003109821", ... }
    fred:  { series_id: "JPNPRINTO01GYSAM" }
```
`transform: derived` uses `formula: "A - B"` evaluated from already-loaded series.

### Inventory cycle module (`analysis/inventory_cycle.py`)

Separate 4-phase model (積み増し加速 / 積み増し / 調整加速 / 調整) using:
`出荷在庫バランス = 出荷YoY% − 在庫YoY%`

Key internals:
- `_CYCLE_WINDOW = 60` months for peak/trough reference
- `_quarterly_series()` samples monthly series at Q-5 → Q0 (offsets 16,13,10,7,4,1)
- `plot_sector_snapshot()` draws per-sector sparklines (Q-5→Q0) inside floating cycle range bars
- `_save_inv_cycle_summary()` writes `data/inv_cycle_summary.json`; reads prior run for `prev_phase` before overwriting
- Sector `cycle_stage: early/mid/late` is set in `config/{country}/manufacturing.yaml` and drives X-axis background shading

### Output files

Per-country run produces:
- `data/{country}/level_momentum.parquet` — monthly Level / Momentum / Stage series
- `data/{country}/cycle_detail.json` — metadata for dashboard
- `data/{country}/stage_timeline.png`
- `data/{country}/inv_cycle_ts.png`, `inv_cycle_sector.png`
- `data/inv_cycle_activity.png` — cross-country BCI/CLI comparison
- `data/inv_cycle_summary.json` — phase + recovery scores (read by report generator)
- `docs/index.html` — self-contained HTML dashboard (base64-embedded PNGs)
- `reports/YYYY-MM.md` — monthly Markdown report

`data/*.parquet` is gitignored. PNGs, JSONs, HTML, and MDs are committed.

### CI/CD

`.github/workflows/monthly_update.yml` runs on the 25th of each month (09:00 UTC).
Steps: `run_cycle.py` × 3 → `generate_report.py` → `generate_dashboard.py` → git push.
Required secrets: `ESTAT_APP_ID`, `FRED_API_KEY`, `ECOS_API_KEY`.
