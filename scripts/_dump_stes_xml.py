import requests, xml.etree.ElementTree as ET

hdrs = {"Accept": "application/vnd.sdmx.structure+json; version=1.0",
        "Accept-Encoding": "identity"}
r = requests.get(
    "https://sdmx.oecd.org/public/rest/dataflow/OECD.SDD.STES",
    headers=hdrs, timeout=20
)

# Raw XML の最初の3000文字を確認
with open("data/stes_raw.xml", "wb") as f:
    f.write(r.content)

# XML 構造を確認
root = ET.fromstring(r.content)
print("Root tag:", root.tag)
print()
# 最初の3レベルだけ表示
def show(el, depth=0, max_depth=3):
    if depth > max_depth:
        return
    attrs = dict(el.attrib)
    print("  " * depth + el.tag.split("}")[-1], attrs)
    for child in list(el)[:5]:
        show(child, depth+1, max_depth)

show(root, 0, 3)
