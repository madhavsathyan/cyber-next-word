import requests, time
from datetime import datetime, timedelta

URL = "https://services.nvd.nist.gov/rest/json/cves/2.0"
FMT = "%Y-%m-%dT%H:%M:%S.000"

start = datetime(2023, 1, 1)
end_limit = datetime(2024, 12, 31)
descriptions = []

while start < end_limit and len(descriptions) < 12000:
    end = min(start + timedelta(days=100), end_limit)   # API max window is 120 days
    params = {
        "pubStartDate": start.strftime(FMT),
        "pubEndDate": end.strftime(FMT),
        "resultsPerPage": 2000,
        "startIndex": 0,
    }
    while True:
        r = requests.get(URL, params=params, timeout=60)
        r.raise_for_status()
        data = r.json()
        for item in data["vulnerabilities"]:
            cve = item["cve"]
            if cve.get("vulnStatus") == "Rejected":
                continue
            for d in cve["descriptions"]:
                if d["lang"] == "en":
                    descriptions.append(d["value"].replace("\n", " "))
        params["startIndex"] += 2000
        print(f"{start.date()} -> {len(descriptions)} descriptions so far")
        time.sleep(7)    # stay under the rate limit
        if params["startIndex"] >= data["totalResults"]:
            break
    start = end

with open("data/raw_descriptions.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(descriptions))
print("Saved", len(descriptions))