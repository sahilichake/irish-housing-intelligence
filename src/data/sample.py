"""Generate a SYNTHETIC dataset in the exact raw PPR format.

Use only for testing the pipeline offline / in CI:
    python -m src.pipeline --sample
Numbers are made up. Never present results from this data as real findings.
"""
import numpy as np
import pandas as pd

from src.config import COUNTIES, RAW_CSV

BASE_PRICE = {  # rough 2024 medians, used only to make the fake data plausible
    "Dublin": 480_000, "Wicklow": 450_000, "Kildare": 390_000, "Meath": 360_000,
    "Cork": 330_000, "Galway": 320_000, "Louth": 300_000, "Limerick": 270_000,
    "Wexford": 270_000, "Kilkenny": 280_000, "Waterford": 260_000, "Westmeath": 250_000,
    "Clare": 250_000, "Kerry": 250_000, "Laois": 250_000, "Carlow": 250_000,
    "Offaly": 220_000, "Tipperary": 220_000, "Sligo": 210_000, "Mayo": 200_000,
    "Cavan": 200_000, "Monaghan": 200_000, "Donegal": 180_000, "Roscommon": 180_000,
    "Longford": 170_000, "Leitrim": 160_000,
}
COUNTY_WEIGHT = {c: (8 if c == "Dublin" else 3 if c in ("Cork", "Kildare", "Meath", "Galway") else 1) for c in COUNTIES}
STREETS = ["Main Street", "Church Road", "Oak Park", "The Avenue", "Riverside", "Seaview",
           "Meadow Lane", "Castle Heights", "Abbey Court", "Green Road"]
ROUTING = {"Dublin": ["D04", "D06", "D08", "D12", "D15", "D24", "A94", "K78"], "Cork": ["T12", "T23", "P31"],
           "Galway": ["H91", "H54"], "Limerick": ["V94", "V35"]}


def year_index(year: float) -> float:
    """Approximate national price index (2024 = 1.0): crash to 2013, then recovery."""
    pts = {2010: 0.62, 2012: 0.47, 2013: 0.45, 2015: 0.58, 2018: 0.75, 2020: 0.78, 2022: 0.90, 2024: 1.0, 2026: 1.10}
    xs, ys = list(pts), list(pts.values())
    return float(np.interp(year, xs, ys))


def generate(n: int = 40_000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    counties = rng.choice(list(COUNTY_WEIGHT), size=n, p=np.array(list(COUNTY_WEIGHT.values())) / sum(COUNTY_WEIGHT.values()))
    dates = pd.to_datetime("2010-01-01") + pd.to_timedelta(rng.integers(0, 365 * 16 + 240, n), unit="D")
    rows = []
    for county, date in zip(counties, dates):
        yr = date.year + date.month / 12
        is_new = rng.random() < 0.17
        is_apt = rng.random() < (0.35 if county == "Dublin" else 0.12)
        size = rng.choice(["", "lt38", "38to125", "gt125"], p=[0.75, 0.02, 0.17, 0.06]) if date.year < 2014 else ""
        price = BASE_PRICE[county] * year_index(yr) * rng.lognormal(0, 0.32)
        price *= 0.75 if is_apt else 1.0
        price *= 1.12 if is_new else 1.0
        price *= {"lt38": 0.55, "gt125": 1.6}.get(size, 1.0)
        district = ""
        if county == "Dublin" and rng.random() < 0.7:
            d = int(rng.choice([1, 2, 4, 6, 7, 8, 9, 11, 12, 15, 18, 24]))
            district = f", Dublin {d}"
            price *= {4: 1.6, 6: 1.5, 18: 1.3, 1: 0.8, 11: 0.85, 24: 0.85}.get(d, 1.0)
        prefix = f"Apt {rng.integers(1, 120)}, " if is_apt else f"{rng.integers(1, 200)} "
        address = f"{prefix}{rng.choice(STREETS)}{district}, Co. {county}"
        eircode = ""
        if date.year >= 2018 and rng.random() < 0.45:
            rk = rng.choice(ROUTING.get(county, ["R" + str(rng.integers(10, 99))]))
            eircode = f"{rk} {''.join(rng.choice(list('ACDEFHKNPRTVWXY0123456789'), 4))}"
        vat_excl = is_new and rng.random() < 0.6
        if vat_excl:
            price /= 1.135
        nfmp = rng.random() < 0.04
        if nfmp:
            price *= 0.5
        desc = "New Dwelling house /Apartment" if is_new else "Second-Hand Dwelling house /Apartment"
        if rng.random() < 0.01:
            desc = "Teach/Árasán Cónaithe Nua" if is_new else "Teach/Árasán Cónaithe Atháimhe"
        size_txt = {"": "", "lt38": "less than 38 sq metres",
                    "38to125": "greater than or equal to 38 sq metres and less than 125 sq metres",
                    "gt125": "greater than 125 sq metres"}[size]
        rows.append({
            "Date of Sale (dd/mm/yyyy)": date.strftime("%d/%m/%Y"),
            "Address": address, "County": county, "Eircode": eircode,
            "Price (€)": f"€{price:,.2f}",
            "Not Full Market Price": "Yes" if nfmp else "No",
            "VAT Exclusive": "Yes" if vat_excl else "No",
            "Description of Property": desc,
            "Property Size Description": size_txt,
        })
    return pd.DataFrame(rows)


def write_sample(n: int = 40_000) -> None:
    df = generate(n)
    # Real PPR file is Windows-1252 encoded; mimic that so the parser is tested properly
    df.to_csv(RAW_CSV, index=False, encoding="cp1252")
    print(f"[sample] Wrote SYNTHETIC data ({len(df):,} rows) to {RAW_CSV}")


if __name__ == "__main__":
    write_sample()
