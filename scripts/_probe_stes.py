import requests
import xml.etree.ElementTree as ET

NS_STRUCT = "http://www.sdmx.org/resources/sdmxml/schemas/v2_1/structure"
NS_COMMON = "http://www.sdmx.org/resources/sdmxml/schemas/v2_1/common"

hdrs = {"Accept": "application/vnd.sdmx.structure+json; version=1.0",
        "Accept-Encoding": "identity"}

r = requests.get(
    "https://sdmx.oecd.org/public/rest/dataflow/OECD.SDD.STES",
    headers=hdrs, timeout=20
)
print("CT:", r.headers.get("Content-Type"))
print("len:", len(r.content))

root = ET.fromstring(r.content)

with open("data/stes_flows.txt", "w", encoding="utf-8") as fout:
    for df in root.iter(f"{{{NS_STRUCT}}}Dataflow"):
        fid = df.get("id", "")
        name = ""
        for el in df.iter(f"{{{NS_COMMON}}}Name"):
            name = el.text or ""
            break
        fout.write(f"{fid} -- {name}\n")

print("Done")
