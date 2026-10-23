import pandas as pd
import pytest

from src.llm.evaluate import results_match
from src.llm.text_to_sql import UnsafeSQLError, extract_sql, validate_sql


@pytest.mark.parametrize("sql", [
    "SELECT county, MEDIAN(price_eur) FROM sales GROUP BY county",
    "WITH x AS (SELECT * FROM sales) SELECT COUNT(*) FROM x;",
    "SELECT * FROM sales WHERE address = 'Drop Lane'",  # keyword inside a string is fine
])
def test_allows_read_only(sql):
    assert validate_sql(sql).startswith("SELECT * FROM (")


@pytest.mark.parametrize("sql", [
    "DROP TABLE sales",
    "SELECT 1; DELETE FROM sales",
    "SELECT * FROM read_csv_auto('/etc/passwd')",
    "COPY sales TO 'out.csv'",
    "ATTACH 'other.db'",
    "",
])
def test_blocks_unsafe(sql):
    with pytest.raises(UnsafeSQLError):
        validate_sql(sql)


def test_extract_sql_from_markdown():
    assert extract_sql("Here:\n```sql\nSELECT 1;\n```") == "SELECT 1;"


def test_results_match_ignores_names_and_extra_cols():
    gold = pd.DataFrame({"m": [350000.0]})
    pred = pd.DataFrame({"county": ["Dublin"], "median_price": [350000.4]})
    assert results_match(gold, pred)
    assert not results_match(gold, pd.DataFrame({"m": [300000.0]}))
