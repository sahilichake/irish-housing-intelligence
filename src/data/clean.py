"""Clean the raw Property Price Register CSV and engineer features."""
import re

import numpy as np
import pandas as pd

from src.config import COUNTIES, MAX_PRICE, MIN_PRICE, RAW_CSV, VAT_RATE

RAW_COLUMNS = [
    "sale_date", "address", "county", "eircode", "price_raw",
    "not_full_market_price", "vat_exclusive", "description", "size_description",
]

APARTMENT_RE = re.compile(r"\b(?:apt|apartment|flat|penthouse|duplex|unit \d+|no\.? ?\d+ block)\b", re.I)
DUBLIN_DISTRICT_RE = re.compile(r"dublin\s*(\d{1,2})\s*(w)?\b", re.I)
EIRCODE_RE = re.compile(r"^([AC-FHKNPRTV-Y]\d{2}|D6W)", re.I)


def read_raw(path=RAW_CSV) -> pd.DataFrame:
    """The PPR file is Windows-1252 encoded and the price column contains a € sign."""
    df = pd.read_csv(path, encoding="cp1252", dtype=str, keep_default_na=False)
    if df.shape[1] != len(RAW_COLUMNS):
        raise ValueError(f"Expected {len(RAW_COLUMNS)} columns, got {df.shape[1]}: {list(df.columns)}")
    df.columns = RAW_COLUMNS
    return df


def parse_price(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.str.replace(r"[^\d.]", "", regex=True), errors="coerce")


def size_band(text: str) -> str:
    t = (text or "").lower()
    if not t:
        return "unknown"
    if "less than 38" in t or "níos lú ná 38" in t:
        return "<38m2"
    if "greater than 125" in t or "níos mó ná 125" in t:
        return ">125m2"
    if "38" in t and "125" in t:
        return "38-125m2"
    return "unknown"


def dublin_district(address: str, county: str) -> str:
    if county != "Dublin":
        return "Outside Dublin"
    m = DUBLIN_DISTRICT_RE.search(address or "")
    if not m:
        return "Co. Dublin (no district)"
    num, west = int(m.group(1)), m.group(2)
    if num < 1 or num > 24:
        return "Co. Dublin (no district)"
    return f"D{num}{'W' if west else ''}"


LOCALITY_SKIP_RE = re.compile(r"^(co\.?|county)\s|^dublin\s*\d|^d\d{1,2}w?$|^\d", re.I)


def locality(address: str, county: str) -> str:
    """Town/area name: the last address part that is not the county, 'Co. X' or 'Dublin 15'."""
    parts = [p.strip() for p in (address or "").split(",") if p.strip()]
    county_l = (county or "").lower()
    for part in reversed(parts[1:] or parts):
        clean_p = re.sub(r"[^a-z' ]", "", part.lower()).strip()
        clean_p = re.sub(r"\s+", " ", clean_p)
        if not clean_p or clean_p == county_l or LOCALITY_SKIP_RE.search(part.strip()):
            continue
        if clean_p in ("co " + county_l, "county " + county_l):
            continue
        return clean_p.upper()
    return "UNKNOWN"


def eircode_routing_key(eircode: str) -> str:
    m = EIRCODE_RE.match((eircode or "").strip())
    return m.group(1).upper() if m else "UNKNOWN"


def clean(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    out["sale_date"] = pd.to_datetime(df["sale_date"], format="%d/%m/%Y", errors="coerce")
    out["address"] = df["address"].str.strip()
    out["county"] = df["county"].str.strip().str.title()
    out["eircode"] = df["eircode"].str.strip().str.upper()

    price = parse_price(df["price_raw"])
    vat_excl = df["vat_exclusive"].str.strip().str.lower().eq("yes")
    out["price_eur"] = np.where(vat_excl, price * (1 + VAT_RATE), price).round(0)

    out["not_full_market_price"] = df["not_full_market_price"].str.strip().str.lower().eq("yes")
    desc = df["description"].str.lower()
    out["is_new"] = desc.str.contains("new|nua", regex=True)
    out["is_apartment"] = out["address"].str.contains(APARTMENT_RE)
    out["size_band"] = df["size_description"].map(size_band)

    out["dublin_district"] = [dublin_district(a, c) for a, c in zip(out["address"], out["county"])]
    out["eircode_routing"] = out["eircode"].map(eircode_routing_key)
    out["locality"] = [locality(a, c) for a, c in zip(out["address"], out["county"])]
    out["has_eircode"] = out["eircode_routing"].ne("UNKNOWN")

    out["year"] = out["sale_date"].dt.year
    out["month"] = out["sale_date"].dt.month
    out["quarter"] = out["sale_date"].dt.quarter
    out["period"] = out["sale_date"].dt.to_period("M").dt.to_timestamp()

    before = len(out)
    out = out.dropna(subset=["sale_date", "price_eur"])
    out = out[out["county"].isin(COUNTIES)]
    out = out[~out["not_full_market_price"]]
    out = out[out["price_eur"].between(MIN_PRICE, MAX_PRICE)]
    out = out.drop_duplicates(subset=["sale_date", "address", "price_eur"])
    out = out.drop(columns=["not_full_market_price"]).reset_index(drop=True)
    # Rare localities (typos, townlands) are grouped so the model only learns well-supported areas
    counts = out.groupby(["county", "locality"])["locality"].transform("size")
    out.loc[counts < 30, "locality"] = "OTHER"
    print(f"[clean] {before:,} raw rows -> {len(out):,} clean rows ({before - len(out):,} removed)")
    return out


def validate(df: pd.DataFrame) -> list[str]:
    """Lightweight data-quality checks. Returns a list of failures (empty = all good)."""
    failures = []
    if df.empty:
        failures.append("dataset is empty")
    if df["sale_date"].isna().any():
        failures.append("null sale dates")
    if not df["price_eur"].between(MIN_PRICE, MAX_PRICE).all():
        failures.append("prices outside allowed range")
    if not df["county"].isin(COUNTIES).all():
        failures.append("unknown county values")
    if df["sale_date"].max() > pd.Timestamp.today() + pd.Timedelta(days=31):
        failures.append("sale dates in the future")
    dup = df.duplicated(subset=["sale_date", "address", "price_eur"]).sum()
    if dup:
        failures.append(f"{dup} duplicate rows")
    return failures
