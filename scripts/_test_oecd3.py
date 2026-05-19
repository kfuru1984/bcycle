"""OECD adapter -- unemployment and exports diagnostics."""
from datetime import date
from bcycle_jp.adapters.oecd import OecdAdapter

a = OecdAdapter()

def report(label, s):
    print(f"{label}")
    print(f"  coverage: {str(s.index[0])[:7]} to {str(s.index[-1])[:7]}  ({len(s)} obs)")
    print(f"  latest:   {s.iloc[-1]:.3f}")
    print(f"  2020-04: {float(s.get(s.index[s.index >= '2020-04'][0], float('nan'))):.3f}" if len(s.loc['2020-04':]) else "  2020-04: n/a")
    print()

# --- Unemployment: try ADJUSTMENT=N (non-SA) for better coverage ---
try:
    s = a.fetch(
        {
            "dataflow": "OECD.SDD.TPS,DSD_LFS@DF_IALFS_UNE_M,1.0",
            "key": "KOR.........",
            "select": {"SEX": "_T", "AGE": "Y_GE15", "FREQ": "M",
                       "ADJUSTMENT": "N", "TRANSFORMATION": "_Z"},
        },
        start=date(1990, 1, 1),
    )
    report("Unemployment (non-SA, Y_GE15, total, M)", s)
except Exception as e:
    print(f"  FAILED: {e}\n")

# --- Exports: add UNIT_MEASURE=GR to select ---
try:
    s2 = a.fetch(
        {
            "dataflow": "OECD.SDD.STES,DSD_KEI@DF_KEI,4.0",
            "key": "KOR.M.EX.GR._T.......?",
            "select": {"MEASURE": "EX", "UNIT_MEASURE": "GR",
                       "ACTIVITY": "_T", "TRANSFORMATION": "GY"},
        },
        start=date(1990, 1, 1),
    )
    report("Exports YoY % (UNIT=GR)", s2)
except Exception as e:
    print(f"  Exports GR FAILED: {e}\n")

# --- Exports: try UNIT_MEASURE=PA ---
try:
    s3 = a.fetch(
        {
            "dataflow": "OECD.SDD.STES,DSD_KEI@DF_KEI,4.0",
            "key": "KOR.M.EX.......",
            "select": {"MEASURE": "EX", "UNIT_MEASURE": "PA",
                       "TRANSFORMATION": "GY"},
        },
        start=date(1990, 1, 1),
    )
    report("Exports YoY % (UNIT=PA)", s3)
except Exception as e:
    print(f"  Exports PA FAILED: {e}\n")
