import re

lines = open("data/raw_descriptions.txt", encoding="utf-8").read().split("\n")
clean = []

for t in lines:
    t = t.lower()
    if t.startswith("** reject") or t.startswith("** dispute"):
        continue
    t = t.split(" note:")[0]                                # drop long "NOTE:" explanations
    t = re.sub(r"cve-\d{4}-\d{4,7}", "<cve>", t)            # CVE IDs
    t = re.sub(r"https?://\S+", "<url>", t)                 # links
    t = re.sub(r"\b\d+(\.\d+)+\b", "<ver>", t)              # versions like 7.2.54.1
    t = re.sub(r"\b0x[0-9a-f]+\b", "<hex>", t)              # hex values
    t = re.sub(r"\b\d+\b", "<num>", t)                      # remaining numbers
    t = re.sub(r"[^a-z0-9<>\s\.\-]", " ", t)                # odd symbols
    t = re.sub(r"\s+", " ", t).strip()
    if 8 <= len(t.split()) <= 60:                           # drop tiny and huge entries
        clean.append(t)

clean = list(dict.fromkeys(clean))                          # remove duplicates
open("data/corpus.txt", "w", encoding="utf-8").write("\n".join(clean))
print("Clean descriptions:", len(clean))