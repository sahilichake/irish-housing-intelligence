"""Feature engineering shared by training, the API and the Streamlit app."""
import numpy as np
import pandas as pd

CATEGORICAL = ["county", "locality", "dublin_district", "eircode_routing", "size_band", "street_type"]
BOOLEAN = ["is_new", "is_apartment", "has_eircode"]
HISTORY = ["prev_rel_price", "years_since_prev", "n_prev_sales",
           "street_rel_price", "street_n_sales", "street_last", "street_recent", "street_days_since",
           "estate_rel_price", "estate_n_sales", "estate_recent"]
TEXT = ["no_number", "n_parts"]
NUMERIC = ["month"] + HISTORY + TEXT
STREET_TYPES = ("ROAD|RD|STREET|ST|AVENUE|AVE|PARK|COURT|CT|GROVE|DRIVE|CRESCENT|CLOSE|LAWN|LAWNS|HEIGHTS|"
                "GARDENS|WAY|VIEW|SQUARE|TERRACE|LANE|MEWS|WOOD|HALL|MANOR|COTTAGE|LODGE|HOUSE|FARM|"
                "VILLAGE|PLACE|GREEN|RISE|DALE|HILL|ESTATE|SITE")
FEATURES = CATEGORICAL + BOOLEAN + NUMERIC


def market_index(df: pd.DataFrame) -> pd.Series:
    """National market level per month, LAGGED so it only uses past sales.

    index[t] = median sale price over months t-3..t-1.
    A valuation at time t can only know the market up to last month, so this
    avoids leaking the target into the features.
    """
    monthly = df.groupby("period")["price_eur"].median().sort_index()
    full = pd.date_range(monthly.index.min(), monthly.index.max() + pd.offsets.MonthBegin(1), freq="MS")
    monthly = monthly.reindex(full).interpolate()
    lagged = monthly.rolling(3, min_periods=1).median().shift(1).bfill()
    lagged.index.name = "period"
    return lagged.rename("market_index")


def address_key(address: pd.Series, county: pd.Series) -> pd.Series:
    norm = (address.str.upper().str.replace(r"[^A-Z0-9 ]", " ", regex=True)
            .str.replace(r"\s+", " ", regex=True).str.strip())
    return county + "|" + norm


STREET_PREFIX_RE = r"^(?:(?:APT|APARTMENT|UNIT|NO|FLAT|HOUSE)\s*\w*\s+)?\d+[A-Z]?\s+"


def street_key(address: pd.Series, county: pd.Series) -> pd.Series:
    """'10 Lismore Rd, Kimmage, Dublin 12' -> 'Dublin|LISMORE RD|KIMMAGE'."""
    parts = address.str.upper().str.replace(r"[^A-Z0-9, ]", " ", regex=True).str.split(",")
    # Drop a leading part that is only a unit number, e.g. "APT 4, OAK PARK, ..." -> "OAK PARK, ..."
    unit_only = parts.str[0].str.strip().str.match(r"^(?:APT|APARTMENT|UNIT|FLAT|NO|HOUSE)?\s*\d+[A-Z]?$")
    parts = parts.where(~unit_only, parts.str[1:])
    first = parts.str[0].fillna("").str.strip().str.replace(STREET_PREFIX_RE, "", regex=True).str.replace(r"\s+", " ", regex=True)
    second = parts.str[1].fillna("").str.strip().str.replace(r"\s+", " ", regex=True)
    return county + "|" + first + "|" + second


def add_sale_history(df: pd.DataFrame, index: pd.Series) -> pd.DataFrame:
    """Repeat-sales features: what the SAME property sold for before (strictly earlier date only).

    prev_rel_price = log(previous price / market index at that time), i.e. how expensive the
    home was relative to the market when it last sold. Same-day sales (bulk/portfolio deals)
    are ignored. Homes never sold before get NaN, which LightGBM handles natively.
    """
    out = df.copy()
    out["_key"] = address_key(out["address"], out["county"])
    out = out.sort_values(["_key", "sale_date"])
    g = out.groupby("_key", sort=False)
    prev_date = g["sale_date"].shift(1)
    prev_price = g["price_eur"].shift(1)
    prev_period = g["period"].shift(1)
    same_day = prev_date.notna() & (prev_date == out["sale_date"])
    prev_idx = prev_period.map(index)
    out["prev_rel_price"] = np.where(same_day | prev_price.isna(), np.nan, np.log(prev_price / prev_idx))
    out["years_since_prev"] = np.where(same_day | prev_date.isna(), np.nan,
                                       (out["sale_date"] - prev_date).dt.days / 365.25)
    out["n_prev_sales"] = g.cumcount().astype(float)

    out = out.drop(columns="_key").sort_index()

    # Street and estate level: price history of EARLIER days only (same-day sales never see each other)
    sk = street_key(out["address"], out["county"])
    rel = np.log(out["price_eur"] / out["period"].map(index))
    st = past_group_stats(sk, out["sale_date"], rel)
    es = past_group_stats(estate_key(sk, out["county"], out["locality"]), out["sale_date"], rel)
    out["street_rel_price"], out["street_n_sales"] = st["mean"], st["n"]
    out["street_last"], out["street_recent"], out["street_days_since"] = st["last"], st["recent"], st["days_since"]
    out["estate_rel_price"], out["estate_n_sales"], out["estate_recent"] = es["mean"], es["n"], es["recent"]
    return add_text_features(out)


def estate_key(street: pd.Series, county: pd.Series, locality: pd.Series) -> pd.Series:
    """Estate/area = the address part after the street; falls back to the town if missing."""
    area = street.str.split("|").str[2].fillna("")
    return county + "|" + area.where(area != "", "TOWN:" + locality.astype(str))


def past_group_stats(key: pd.Series, date: pd.Series, rel: pd.Series) -> pd.DataFrame:
    """For each sale, stats of its group (street/estate) over strictly EARLIER days.

    Sales are first aggregated per group per day, so sales on the same day never see each other.
    mean/n = all earlier sales (mean needs >= 3), last = previous sale day's mean,
    recent = mean of the last 5 earlier sale days, days_since = days since the previous sale day.
    """
    day = (pd.DataFrame({"k": key.values, "date": date.values, "rel": rel.values})
           .groupby(["k", "date"], sort=True)["rel"].agg(["sum", "count", "mean"]).reset_index())
    g = day.groupby("k", sort=False)
    cs, cc = g["sum"].cumsum() - day["sum"], g["count"].cumsum() - day["count"]
    day["mean_"] = np.where(cc >= 3, cs / cc.where(cc > 0), np.nan)
    day["n"] = cc.astype(float)
    day["last"] = g["mean"].shift(1)
    day["recent"] = g["mean"].transform(lambda x: x.shift(1).rolling(5, min_periods=2).mean())
    day["days_since"] = (day["date"] - g["date"].shift(1)).dt.days.astype(float)
    m = pd.DataFrame({"k": key.values, "date": date.values}).merge(
        day[["k", "date", "mean_", "n", "last", "recent", "days_since"]], on=["k", "date"], how="left")
    m.index = key.index
    return m.rename(columns={"mean_": "mean"})


def add_text_features(df: pd.DataFrame) -> pd.DataFrame:
    """Clues from the address itself, e.g. a named house with no number is often a one-off rural home."""
    out = df.copy()
    parts = out["address"].fillna("").str.upper().str.replace(r"[^A-Z0-9, ]", " ", regex=True).str.split(",")
    first = parts.str[0].str.strip()
    out["no_number"] = (~first.str.contains(r"\d")).astype(float)
    out["n_parts"] = parts.str.len().astype(float)
    out["street_type"] = first.str.extract(rf"\b({STREET_TYPES})\b")[0].fillna("NONE")
    return out


def build_matrix(df: pd.DataFrame, categories: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Return model-ready X. Pass `categories` from training at inference time."""
    X = df[FEATURES].copy()
    for b in BOOLEAN:
        X[b] = X[b].astype(int)
    categories = categories or {c: sorted(X[c].astype(str).unique()) for c in CATEGORICAL}
    for c in CATEGORICAL:
        vals = X[c].astype(str)
        X[c] = pd.Categorical(vals.where(vals.isin(categories[c])), categories=categories[c])  # unseen -> NaN
    X["month"] = X["month"].astype(int)
    for c in HISTORY + TEXT:
        X[c] = pd.to_numeric(X[c], errors="coerce").astype(float)
    return X, categories


def to_log_relative(price: pd.Series, index: pd.Series) -> np.ndarray:
    return np.log(price.values) - np.log(index.values)


def from_log_relative(pred: np.ndarray, index: pd.Series | np.ndarray) -> np.ndarray:
    return np.exp(pred + np.log(np.asarray(index)))
