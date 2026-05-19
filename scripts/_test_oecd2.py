"""OECD adapter smoke test — CPI, unemployment, exports, consumer confidence."""
from datetime import date
from bcycle_jp.adapters.oecd import OecdAdapter

a = OecdAdapter()

def report(label, s):
    print(f"{label}")
    print(f"  coverage: {str(s.index[0])[:7]} to {str(s.index[-1])[:7]}  ({len(s)} obs)")
    print(f"  latest:   {s.iloc[-1]:.3f}")
    print()

# 1. Core CPI YoY (ex food & energy)
s1 = a.fetch(
    {
        "dataflow": "OECD.SDD.TPS,DSD_PRICES@DF_PRICES_N_TXCP01_NRG,1.0",
        "key": "KOR.M.N.CPI.PA._TXCP01_NRG.N.GY",
    },
    start=date(1990, 1, 1),
)
report("Core CPI YoY (%)", s1)

# 2. Unemployment SA (15+, total, monthly)
s2 = a.fetch(
    {
        "dataflow": "OECD.SDD.TPS,DSD_LFS@DF_IALFS_UNE_M,1.0",
        "key": "KOR.........",
        "select": {"MEASURE": "UNE_LF_M", "ADJUSTMENT": "Y",
                   "SEX": "_T", "AGE": "Y_GE15", "FREQ": "M"},
    },
    start=date(1990, 1, 1),
)
report("Unemployment SA % (Y_GE15, total, M)", s2)

# 3. Exports YoY from KEI
s3 = a.fetch(
    {
        "dataflow": "OECD.SDD.STES,DSD_KEI@DF_KEI,4.0",
        "key": "KOR.M.EX.......",
        "select": {"TRANSFORMATION": "GY"},
    },
    start=date(1990, 1, 1),
)
report("Exports YoY % (KEI)", s3)

# 4. Consumer Confidence from KEI
s4 = a.fetch(
    {
        "dataflow": "OECD.SDD.STES,DSD_KEI@DF_KEI,4.0",
        "key": "KOR.M.CCICP.......",
        "select": {"TRANSFORMATION": "_Z"},
    },
    start=date(1990, 1, 1),
)
report("Consumer Confidence (CCICP, level)", s4)
