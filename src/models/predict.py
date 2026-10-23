"""Load the trained model and price a single property (with SHAP explanation)."""
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from src.config import PRICE_MODEL_PATH
from src.data.clean import dublin_district, eircode_routing_key
from src.data.database import query
from src.models.features import (FEATURES, HISTORY, add_text_features, address_key, build_matrix,
                                 estate_key, street_key)


@lru_cache(maxsize=1)
def load_bundle() -> dict:
    if not PRICE_MODEL_PATH.exists():
        raise FileNotFoundError("Model not trained yet. Run: python -m src.pipeline")
    return joblib.load(PRICE_MODEL_PATH)


def _group_now(sub: pd.DataFrame) -> dict:
    """Stats of a street/estate as of today, matching the training features (all sales are in the past)."""
    if sub.empty:
        return {"mean": np.nan, "n": 0.0, "last": np.nan, "recent": np.nan, "days_since": np.nan}
    day = sub.groupby("sale_date")["_rel"].mean().sort_index()
    return {"mean": float(sub["_rel"].mean()) if len(sub) >= 3 else np.nan, "n": float(len(sub)),
            "last": float(day.iloc[-1]), "recent": float(day.iloc[-5:].mean()) if len(day) >= 2 else np.nan,
            "days_since": float((pd.Timestamp.today() - day.index[-1]).days)}


def sale_history(address: str, county: str, town: str = "UNKNOWN") -> dict:
    """Look up earlier sales of this exact address, its street and its estate (only if an address is given)."""
    out = {k: np.nan for k in HISTORY}
    out.update(n_prev_sales=0.0, street_n_sales=0.0, estate_n_sales=0.0, found_prev_sale=None)
    if not (address or "").strip():
        return out
    idx = load_bundle()["index_series"]
    a, c = pd.Series([address]), pd.Series([county])
    skey, akey = street_key(a, c).iloc[0], address_key(a, c).iloc[0]
    ekey = estate_key(pd.Series([skey]), c, pd.Series([(town or "UNKNOWN").upper()])).iloc[0]
    hist = query("SELECT address, county, locality, sale_date, price_eur, period FROM sales WHERE county = ?", [county])
    hist["_s"] = street_key(hist["address"], hist["county"])
    hist["_k"] = address_key(hist["address"], hist["county"])
    hist["_e"] = estate_key(hist["_s"], hist["county"], hist["locality"])
    hist["_rel"] = np.log(hist["price_eur"] / hist["period"].map(idx).fillna(idx.iloc[-1]))

    st, es = _group_now(hist[hist["_s"] == skey]), _group_now(hist[hist["_e"] == ekey])
    out.update(street_rel_price=st["mean"], street_n_sales=st["n"], street_last=st["last"],
               street_recent=st["recent"], street_days_since=st["days_since"],
               estate_rel_price=es["mean"], estate_n_sales=es["n"], estate_recent=es["recent"])
    own = hist[hist["_k"] == akey].sort_values("sale_date")
    if len(own):
        last = own.iloc[-1]
        out.update(prev_rel_price=float(last["_rel"]), n_prev_sales=float(len(own)),
                   years_since_prev=(pd.Timestamp.today() - last["sale_date"]).days / 365.25,
                   found_prev_sale={"date": f"{last['sale_date']:%Y-%m-%d}", "price_eur": float(last["price_eur"])})
    return out


def make_row(county: str, is_new: bool, is_apartment: bool, size_band: str = "unknown",
             eircode: str = "", dublin_area: str | None = None, month: int = 6,
             town: str = "UNKNOWN", address: str = "") -> pd.DataFrame:
    district = dublin_area or dublin_district(address or "", county)
    routing = eircode_routing_key(eircode)
    hist = sale_history(address, county, town)
    row = pd.DataFrame([{
        "address": address or "", "county": county, "locality": (town or "UNKNOWN").upper(),
        "dublin_district": district, "eircode_routing": routing, "size_band": size_band,
        "is_new": is_new, "is_apartment": is_apartment, "has_eircode": routing != "UNKNOWN", "month": month,
        **{k: v for k, v in hist.items() if k != "found_prev_sale"},
    }])
    row = add_text_features(row)
    if not (address or "").strip():  # no address: text clues are unknown, not "no number"
        row[["no_number", "n_parts"]] = np.nan
        row["street_type"] = "UNKNOWN"
    row.attrs["found_prev_sale"] = hist["found_prev_sale"]
    row.attrs["street_n_sales"] = hist["street_n_sales"]
    return row


def predict(row: pd.DataFrame) -> dict:
    b = load_bundle()
    X, _ = build_matrix(row, b["categories"])
    log_rel = b["booster"].predict(X)[0]
    contrib = b["booster"].predict(X, pred_contrib=True)[0]
    price = float(np.exp(log_rel) * b["latest_index"])
    # Each SHAP value is additive in log space -> convert to an approximate % effect
    effects = {f: float((np.exp(v) - 1) * 100) for f, v in zip(FEATURES, contrib[:-1])}
    return {
        "predicted_price_eur": round(price, -3),
        "market_level_eur": round(b["latest_index"], -3),
        "trained_until": b["trained_until"],
        "previous_sale": row.attrs.get("found_prev_sale"),
        "street_sales_used": int(row.attrs.get("street_n_sales") or 0),
        "feature_effects_pct": dict(sorted(effects.items(), key=lambda kv: -abs(kv[1]))),
    }
