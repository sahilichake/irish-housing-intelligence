import pandas as pd
import pytest

from src.data.clean import clean, dublin_district, eircode_routing_key, size_band, validate


def raw(**over):
    row = {
        "sale_date": "15/06/2023", "address": "Apt 4, Oak Park, Dublin 6W", "county": "Dublin",
        "eircode": "D6W AB12", "price_raw": "€300,000.00", "not_full_market_price": "No",
        "vat_exclusive": "No", "description": "Second-Hand Dwelling house /Apartment", "size_description": "",
    }
    row.update(over)
    return pd.DataFrame([row])


def test_basic_parsing():
    out = clean(raw())
    r = out.iloc[0]
    assert r.price_eur == 300_000
    assert r.dublin_district == "D6W"
    assert r.eircode_routing == "D6W"
    assert bool(r.is_apartment) and not bool(r.is_new)
    assert r.year == 2023 and r.month == 6


def test_vat_added_for_vat_exclusive_new_builds():
    out = clean(raw(vat_exclusive="Yes", description="New Dwelling house /Apartment"))
    assert out.iloc[0].price_eur == pytest.approx(340_500)
    assert bool(out.iloc[0].is_new)


def test_irish_language_new_build_detected():
    assert bool(clean(raw(description="Teach/Árasán Cónaithe Nua")).iloc[0].is_new)


def test_filters_non_market_and_outliers():
    df = pd.concat([raw(not_full_market_price="Yes"), raw(price_raw="€5,000.00"), raw(county="Narnia")])
    assert clean(df).empty


@pytest.mark.parametrize("addr,county,expected", [
    ("1 Main St, Dublin 4", "Dublin", "D4"),
    ("1 Main St, Dublin 24", "Dublin", "D24"),
    ("1 Main St, Swords, Co Dublin", "Dublin", "Co. Dublin (no district)"),
    ("1 Main St, Cork", "Cork", "Outside Dublin"),
])
def test_dublin_district(addr, county, expected):
    assert dublin_district(addr, county) == expected


def test_eircode_and_size():
    assert eircode_routing_key("t12 x8y9") == "T12"
    assert eircode_routing_key("") == "UNKNOWN"
    assert size_band("greater than 125 sq metres") == ">125m2"
    assert size_band("") == "unknown"


def test_validate_passes_on_clean_data():
    assert validate(clean(raw())) == []
