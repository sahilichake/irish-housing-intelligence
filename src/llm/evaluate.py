"""Evaluate text-to-SQL accuracy against hand-written gold queries.

Metric = execution accuracy: the generated query's result must contain the same
values as the gold query's result (column names and extra columns are ignored).

Run:  python -m src.llm.evaluate            (uses LLM_PROVIDER from .env)
Add more questions to EVAL_SET — aim for 50 for your README.
"""
import json
import math
import time

import pandas as pd

from src.config import LLM_EVAL_PATH
from src.llm.text_to_sql import generate_sql, run_sql

LATEST = "(SELECT MAX(year) FROM sales)"
EVAL_SET = [
    ("What was the median house price in Dublin in 2019?",
     "SELECT ROUND(MEDIAN(price_eur),0) FROM sales WHERE county='Dublin' AND year=2019"),
    ("How many properties were sold in Cork in 2021?",
     "SELECT COUNT(*) FROM sales WHERE county='Cork' AND year=2021"),
    ("Which county had the highest median price in 2022?",
     "SELECT county FROM sales WHERE year=2022 GROUP BY county ORDER BY MEDIAN(price_eur) DESC LIMIT 1"),
    ("Which county had the lowest median price in 2015?",
     "SELECT county FROM sales WHERE year=2015 GROUP BY county ORDER BY MEDIAN(price_eur) ASC LIMIT 1"),
    ("What is the total number of sales in the whole dataset?",
     "SELECT COUNT(*) FROM sales"),
    ("What was the median price of new builds in Kildare in 2020?",
     "SELECT ROUND(MEDIAN(price_eur),0) FROM sales WHERE county='Kildare' AND is_new AND year=2020"),
    ("What was the median price of second-hand homes in Galway in 2018?",
     "SELECT ROUND(MEDIAN(price_eur),0) FROM sales WHERE county='Galway' AND NOT is_new AND year=2018"),
    ("List the top 3 counties by number of sales in 2023",
     "SELECT county FROM sales WHERE year=2023 GROUP BY county ORDER BY COUNT(*) DESC LIMIT 3"),
    ("What was the most expensive sale ever recorded in Mayo?",
     "SELECT MAX(price_eur) FROM sales WHERE county='Mayo'"),
    ("How many apartments were sold in Dublin in 2021?",
     "SELECT COUNT(*) FROM sales WHERE county='Dublin' AND is_apartment AND year=2021"),
    ("What was the median price nationally in 2013?",
     "SELECT ROUND(MEDIAN(price_eur),0) FROM sales WHERE year=2013"),
    ("Show the median price in Limerick for each year",
     "SELECT year, ROUND(MEDIAN(price_eur),0) FROM sales WHERE county='Limerick' GROUP BY year ORDER BY year"),
    ("What share of sales in 2022 were new builds, as a percentage?",
     "SELECT ROUND(100.0*AVG(CASE WHEN is_new THEN 1 ELSE 0 END),1) FROM sales WHERE year=2022"),
    ("Which Dublin postal district had the highest median price in 2021?",
     "SELECT dublin_district FROM sales WHERE county='Dublin' AND year=2021 AND dublin_district LIKE 'D%' "
     "GROUP BY dublin_district ORDER BY MEDIAN(price_eur) DESC LIMIT 1"),
    ("How many sales were there in Wicklow in the latest year in the data?",
     f"SELECT COUNT(*) FROM sales WHERE county='Wicklow' AND year={LATEST}"),
    ("What was the mean sale price in Louth in 2017?",
     "SELECT ROUND(AVG(price_eur),0) FROM sales WHERE county='Louth' AND year=2017"),
    ("How many sales above 1 million euro were there in 2022?",
     "SELECT COUNT(*) FROM sales WHERE price_eur > 1000000 AND year=2022"),
    ("Which month of 2021 had the most sales nationally? Give the month number.",
     "SELECT month FROM sales WHERE year=2021 GROUP BY month ORDER BY COUNT(*) DESC LIMIT 1"),
    ("What was the median price in Meath in 2016?",
     "SELECT ROUND(MEDIAN(price_eur),0) FROM sales WHERE county='Meath' AND year=2016"),
    ("How many counties had a median price above 300000 euro in 2023?",
     "SELECT COUNT(*) FROM (SELECT county FROM sales WHERE year=2023 GROUP BY county HAVING MEDIAN(price_eur) > 300000)"),
]


def _norm(v):
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return None
        return round(float(v), 0)
    return str(v).strip().lower()


def results_match(gold: pd.DataFrame, pred: pd.DataFrame) -> bool:
    if len(gold) != len(pred):
        return False
    pred_cols = [sorted(map(_norm, pred[c].tolist()), key=str) for c in pred.columns]
    for c in gold.columns:
        g = sorted(map(_norm, gold[c].tolist()), key=str)
        ok = any(
            len(p) == len(g) and all(
                (a == b) or (isinstance(a, float) and isinstance(b, float) and abs(a - b) <= max(1.0, 0.005 * abs(a)))
                for a, b in zip(g, p))
            for p in pred_cols)
        if not ok:
            return False
    return True


def evaluate(provider: str | None = None) -> dict:
    rows, correct = [], 0
    for q, gold_sql in EVAL_SET:
        gold = run_sql(gold_sql)
        t0 = time.time()
        try:
            sql = generate_sql(q, provider)
            ok = results_match(gold, run_sql(sql))
            err = None
        except Exception as exc:  # noqa: BLE001
            sql, ok, err = None, False, str(exc)[:200]
        correct += ok
        rows.append({"question": q, "correct": ok, "sql": sql, "error": err, "seconds": round(time.time() - t0, 2)})
        print(f"[{'PASS' if ok else 'FAIL'}] {q}")
    report = {"n": len(EVAL_SET), "correct": correct,
              "execution_accuracy_pct": round(100 * correct / len(EVAL_SET), 1), "details": rows}
    LLM_EVAL_PATH.write_text(json.dumps(report, indent=2))
    print(f"\nExecution accuracy: {report['execution_accuracy_pct']}% ({correct}/{len(EVAL_SET)})")
    return report


if __name__ == "__main__":
    evaluate()
