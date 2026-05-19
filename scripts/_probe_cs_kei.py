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
    print(f"  Series count: {len(ds.get('series',{}))}")
    print()

# Consumer confidence / opinion surveys for KOR
probe(
    "DF_CS KOR",
    "https://sdmx.oecd.org/public/rest/data/OECD.SDD.STES,DSD_STES@DF_CS,1.0/KOR.......?startPeriod=2023-01&endPeriod=2023-03"
)

# KEI for KOR — key short-term indicators
probe(
    "DF_KEI KOR",
    "https://sdmx.oecd.org/public/rest/data/OECD.SDD.STES,DSD_KEI@DF_KEI,1.0/KOR.......?startPeriod=2023-01&endPeriod=2023-03"
)

# INDSERV for KOR
probe(
    "DF_INDSERV KOR",
    "https://sdmx.oecd.org/public/rest/data/OECD.SDD.STES,DSD_STES@DF_INDSERV,1.0/KOR.......?startPeriod=2023-01&endPeriod=2023-03"
)
