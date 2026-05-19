"""OECD アダプタ 動作確認スクリプト。"""
from datetime import date
from bcycle_jp.adapters.oecd import OecdAdapter

a = OecdAdapter()
print("is_available:", a.is_available())

# 1. Core CPI (KOR) — 完全 key、1系列確定
print("\n--- Core CPI ---")
s = a.fetch(
    {
        "dataflow": "OECD.SDD.TPS,DSD_PRICES@DF_PRICES_N_TXCP01_NRG,1.0",
        "key": "KOR.M.N.CPI.PA._TXCP01_NRG.N.GY",
    },
    start=date(2020, 1, 1),
)
print(f"  coverage: {s.index[0].strftime('%Y-%m')} to {s.index[-1].strftime('%Y-%m')}  ({len(s)} obs)")
print(f"  latest: {s.iloc[-1]:.3f}%")
print(f"  2020-01..03:\n{s.loc['2020-01':'2020-03']}")

# 2. 失業率 (KOR) — ワイルドカード + select
print("\n--- Unemployment (SA, 15+, Total, Monthly) ---")
s2 = a.fetch(
    {
        "dataflow": "OECD.SDD.TPS,DSD_LFS@DF_IALFS_UNE_M,1.0",
        "key": "KOR.........",
        "select": {
            "MEASURE":    "UNE_LF_M",
            "ADJUSTMENT": "Y",
            "SEX":        "_T",
            "AGE":        "Y_GE15",
            "FREQ":       "M",
        },
    },
    start=date(2020, 1, 1),
)
print(f"  coverage: {s2.index[0].strftime('%Y-%m')} to {s2.index[-1].strftime('%Y-%m')}  ({len(s2)} obs)")
print(f"  latest: {s2.iloc[-1]:.3f}%")
