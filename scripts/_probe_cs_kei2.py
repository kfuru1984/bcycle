import requests, json

hdrs = {"Accept": "application/vnd.sdmx.data+json;version=2.0"}

def probe(label, url):
    r = requests.get(url, headers=hdrs, timeout=20)
    print(f"{label} [{r.status_code}]")
    if r.status_code != 200:
        print(" ", r.text[:200])
        return
    body = r.json()
    structs = body.get("data", {}).get("structures", [])
    if structs:
        dims = structs[0].get("dimensions", {})
        print("  Series dims:")
        for d in dims.get("series", []):
            vals = [v["id"] for v in d.get("values", [])[:8]]
            print(f"    [{d.get('keyPosition',0)}] {d['id']}: {vals}")
        for d in dims.get("observation", []):
            vals = [v["id"] for v in d.get("values", [])[:4]]
            print(f"    OBS {d['id']}: {vals}")
    ds = body["data"]["dataSets"][0]
    n = len(ds.get("series", {}))
    print(f"  Series count: {n}")
    # first series sample
    for k, v in list(ds.get("series", {}).items())[:2]:
        obs_sample = list(v.get("observations", {}).items())[:2]
        print(f"  key={k}  obs={obs_sample}")
    print()

# Consumer opinion surveys v4.0
probe("DF_CS KOR",
      "https://sdmx.oecd.org/public/rest/data/OECD.SDD.STES,DSD_STES@DF_CS,4.0/KOR.......?startPeriod=2023-01&endPeriod=2023-03")

# Key short-term indicators v4.0
probe("DF_KEI KOR",
      "https://sdmx.oecd.org/public/rest/data/OECD.SDD.STES,DSD_KEI@DF_KEI,4.0/KOR.......?startPeriod=2023-01&endPeriod=2023-03")

# Production/sales v4.0
probe("DF_INDSERV KOR",
      "https://sdmx.oecd.org/public/rest/data/OECD.SDD.STES,DSD_STES@DF_INDSERV,4.0/KOR.......?startPeriod=2023-01&endPeriod=2023-03")
