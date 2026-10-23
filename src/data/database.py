"""DuckDB storage: one clean `sales` table plus analysis views."""
import duckdb
import pandas as pd

from src.config import DB_PATH

SCHEMA_DESCRIPTION = """
Table: sales  (one row per residential property sale in Ireland, from the Property Price Register)
  sale_date        DATE     date of sale
  address          VARCHAR  full address
  county           VARCHAR  one of the 26 counties, e.g. 'Dublin', 'Cork', 'Galway'
  eircode          VARCHAR  postcode, often empty
  price_eur        DOUBLE   sale price in euro (VAT included)
  is_new           BOOLEAN  true = new build, false = second-hand
  is_apartment     BOOLEAN  true if the address looks like an apartment
  size_band        VARCHAR  '<38m2', '38-125m2', '>125m2' or 'unknown' (mostly unknown)
  locality         VARCHAR  town/area in UPPER CASE, e.g. 'GREYSTONES', 'LUCAN', 'DOUGLAS'; 'OTHER' or 'UNKNOWN' if rare/missing
  dublin_district  VARCHAR  e.g. 'D4', 'D6W', 'Co. Dublin (no district)', 'Outside Dublin'
  eircode_routing  VARCHAR  first 3 chars of Eircode e.g. 'D04', 'T12', or 'UNKNOWN'
  has_eircode      BOOLEAN
  year             INTEGER
  month            INTEGER
  quarter          INTEGER
  period           TIMESTAMP first day of the month of sale

View: monthly_county (county, period, n_sales, median_price, mean_price)
"""


def connect(read_only: bool = False) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(DB_PATH), read_only=read_only)


def write_sales(df: pd.DataFrame, source: str = "ppr") -> None:
    con = connect()
    con.register("df_view", df)
    con.execute("CREATE OR REPLACE TABLE sales AS SELECT * FROM df_view")
    con.execute("CREATE OR REPLACE TABLE metadata AS SELECT ? AS source, current_timestamp AS built_at", [source])
    con.execute("""
        CREATE OR REPLACE VIEW monthly_county AS
        SELECT county, period,
               COUNT(*)                 AS n_sales,
               MEDIAN(price_eur)        AS median_price,
               AVG(price_eur)           AS mean_price
        FROM sales GROUP BY county, period
    """)
    n = con.execute("SELECT COUNT(*) FROM sales").fetchone()[0]
    con.close()
    print(f"[db] Wrote {n:,} rows to {DB_PATH}")


def query(sql: str, params=None) -> pd.DataFrame:
    con = connect(read_only=True)
    try:
        return con.execute(sql, params or []).df()
    finally:
        con.close()


def data_source() -> str:
    """'ppr' for real data, 'sample' for synthetic data."""
    try:
        return query("SELECT source FROM metadata").iloc[0, 0]
    except Exception:  # noqa: BLE001
        return "unknown"
