"""Natural-language questions -> safe DuckDB SQL -> answer.

Supports three providers (set LLM_PROVIDER in .env):
  anthropic  needs ANTHROPIC_API_KEY
  openai     needs OPENAI_API_KEY
  ollama     free + local: install Ollama, run `ollama pull llama3.1`
"""
import os
import re

import pandas as pd
import requests

from src import config
from src.data.database import SCHEMA_DESCRIPTION, query

SYSTEM_PROMPT = f"""You are a senior data analyst who writes DuckDB SQL about the Irish housing market.
{SCHEMA_DESCRIPTION}
Rules:
- Return ONE read-only query (SELECT or WITH) inside a ```sql code block, nothing else.
- Use MEDIAN(price_eur) for "typical"/"average house price" unless the user says mean.
- County names are title case, e.g. county = 'Cork'.
- Round money to whole euro with ROUND(x, 0). Add ORDER BY for rankings.
- For "this year"/"latest" use the most recent year in the data: (SELECT MAX(year) FROM sales).
- Never invent columns that are not in the schema.

Examples:
Q: What was the median house price in Galway in 2023?
```sql
SELECT ROUND(MEDIAN(price_eur), 0) AS median_price FROM sales WHERE county = 'Galway' AND year = 2023;
```
Q: Top 5 counties by number of sales in 2022
```sql
SELECT county, COUNT(*) AS n_sales FROM sales WHERE year = 2022 GROUP BY county ORDER BY n_sales DESC LIMIT 5;
```
"""

FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|create|alter|attach|detach|copy|pragma|install|load|export|import|"
    r"call|truncate|vacuum|checkpoint|read_csv\w*|read_parquet|read_json\w*|glob|"
    r"read_text|read_blob|httpfs|getenv)\b", re.I)


class UnsafeSQLError(ValueError):
    pass


def extract_sql(text: str) -> str:
    m = re.search(r"```(?:sql)?\s*(.*?)```", text, re.S | re.I)
    return (m.group(1) if m else text).strip()


def validate_sql(sql: str, max_rows: int = 1000) -> str:
    """Guardrails: single read-only statement, no file/network access, row-limited."""
    sql = sql.strip().rstrip(";").strip()
    if not sql:
        raise UnsafeSQLError("Empty query")
    if ";" in sql:
        raise UnsafeSQLError("Only a single statement is allowed")
    if not re.match(r"^(select|with)\b", sql, re.I):
        raise UnsafeSQLError("Only SELECT / WITH queries are allowed")
    no_strings = re.sub(r"'[^']*'", "''", sql)  # don't flag words inside string literals
    if m := FORBIDDEN.search(no_strings):
        raise UnsafeSQLError(f"Forbidden keyword: {m.group(0)}")
    return f"SELECT * FROM ({sql}) AS q LIMIT {max_rows}"


def call_llm(system: str, user: str, provider: str | None = None) -> str:
    provider = (provider or os.getenv("LLM_PROVIDER") or config.LLM_PROVIDER).lower()
    if provider == "anthropic":
        key = os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("Set ANTHROPIC_API_KEY in your .env file")
        r = requests.post("https://api.anthropic.com/v1/messages", timeout=60, headers={
            "x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
            json={"model": os.getenv("ANTHROPIC_MODEL", config.ANTHROPIC_MODEL), "max_tokens": 1024,
                  "temperature": 0, "system": system, "messages": [{"role": "user", "content": user}]})
        r.raise_for_status()
        return r.json()["content"][0]["text"]
    if provider == "openai":
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("Set OPENAI_API_KEY in your .env file")
        r = requests.post("https://api.openai.com/v1/chat/completions", timeout=60,
                          headers={"Authorization": f"Bearer {key}"},
                          json={"model": os.getenv("OPENAI_MODEL", config.OPENAI_MODEL), "temperature": 0,
                                "messages": [{"role": "system", "content": system},
                                             {"role": "user", "content": user}]})
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    if provider == "ollama":
        model = os.getenv("OLLAMA_MODEL", config.OLLAMA_MODEL)
        try:
            r = requests.post(f"{os.getenv('OLLAMA_URL', config.OLLAMA_URL)}/api/chat", timeout=300,
                              json={"model": model, "stream": False, "options": {"temperature": 0},
                                    "messages": [{"role": "system", "content": system},
                                                 {"role": "user", "content": user}]})
        except requests.exceptions.ConnectionError:
            raise RuntimeError("Ollama is not running. Install it from https://ollama.com, open the Ollama app, "
                               f"then run:  ollama pull {model}") from None
        if r.status_code == 404:
            raise RuntimeError(f"Ollama model '{model}' is not downloaded yet. Run:  ollama pull {model}")
        r.raise_for_status()
        return r.json()["message"]["content"]
    raise ValueError(f"Unknown LLM_PROVIDER: {provider}")


def generate_sql(question: str, provider: str | None = None) -> str:
    return extract_sql(call_llm(SYSTEM_PROMPT, f"Q: {question}", provider))


def run_sql(sql: str) -> pd.DataFrame:
    return query(validate_sql(sql))


def ask(question: str, provider: str | None = None, summarise: bool = True) -> dict:
    """Full pipeline with one self-correction retry if the SQL fails."""
    sql = generate_sql(question, provider)
    try:
        df = run_sql(sql)
    except UnsafeSQLError:
        raise
    except Exception as exc:  # noqa: BLE001 - give the model one chance to fix its SQL
        fix_prompt = f"Q: {question}\nYour previous SQL:\n```sql\n{sql}\n```\nfailed with: {exc}\nReturn a corrected query."
        sql = extract_sql(call_llm(SYSTEM_PROMPT, fix_prompt, provider))
        df = run_sql(sql)

    answer = None
    if summarise:
        preview = df.head(30).to_csv(index=False) if not df.empty else "(no rows)"
        answer = call_llm(
            "You explain query results about Irish house prices in 2-3 plain sentences. "
            "Only state facts present in the result table. Format euro amounts like €350,000.",
            f"Question: {question}\nResult:\n{preview}", provider)
    return {"question": question, "sql": sql, "result": df, "answer": answer}
