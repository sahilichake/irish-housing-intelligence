# 🏠 Irish Housing Market Intelligence

An end-to-end machine-learning platform built on **every residential property sale in Ireland since 2010**
(Property Price Register). It cleans and validates the data, values properties with an explainable LightGBM
model, forecasts county prices 12 months ahead, and lets you **ask questions in plain English** through a
guarded LLM text-to-SQL layer, all in a Streamlit app.

> **Live demo:** _add your Streamlit / Hugging Face link here_ · **Demo video:** _add GIF here_

## Features

| Page | What it does |
|---|---|
| **Market overview** | National and county trends, biggest movers, Dublin postal-district league table |
| **Price estimator** | Values a property at today's market level, explains it with SHAP, shows comparable sales |
| **Forecasts** | 12-month county forecasts with 80% intervals; the model is chosen per county by backtest |
| **Ask the data (AI)** | Natural language → SQL → answer, with read-only guardrails and visible SQL |
| **Model card** | Test metrics against baselines, feature importance, error by county, LLM eval accuracy |

## Architecture

```
propertypriceregister.ie ──► download ──► clean + validate ──► DuckDB ──┬─► LightGBM price model ──► FastAPI /predict
     (PPR-ALL.csv)                       (VAT fix, outliers,           ├─► county forecasting (naive / seasonal / damped Holt)
                                          districts, Eircodes)         ├─► LLM text-to-SQL (+ guardrails, eval set)
                                                                       └─► Streamlit app
GitHub Actions: tests on every push · monthly retrain on fresh data
```

## Results  _(fill in after running on the real data)_

| Model (test = most recent 12 months) | MAE | MAPE | Within 10% |
|---|---|---|---|
| Baseline: county median | € | % | % |
| Baseline: county × market index | € | % | % |
| **LightGBM** | **€** | **%** | **%** |

Text-to-SQL execution accuracy: **__%** on 20 benchmark questions (`python -m src.llm.evaluate`).

## Quick start

Needs **Python 3.10+**.

```bash
# 1. Set up
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Build data + models (downloads ~25 MB, takes 1–3 minutes)
python -m src.pipeline

# 3. Launch the app
streamlit run app/streamlit_app.py
```

**If the download fails:** go to propertypriceregister.ie → Download → *All*, save `PPR-ALL.zip` into
`data/raw/`, and run step 2 again. To test everything offline, run `python -m src.pipeline --sample`.
That uses **synthetic** data, so don't quote its numbers.

**AI page:** copy `.env.example` to `.env` and add an Anthropic or OpenAI key. For a free local option,
install [Ollama](https://ollama.com), run `ollama pull llama3.1`, and set `LLM_PROVIDER=ollama`.

**Other commands**

```bash
uvicorn api.main:app --reload        # REST API → http://127.0.0.1:8000/docs
pytest -q                            # tests
python -m src.llm.evaluate           # benchmark the LLM layer (writes models/llm_eval.json)
docker build -t irish-housing . && docker run -p 8501:8501 irish-housing
```

## Method

- **Cleaning:** decodes the Windows-1252 file, adds 13.5% VAT back to new builds reported ex-VAT, removes
  *Not Full Market Price* sales and prices outside €30k–€5m, de-duplicates, and extracts Dublin postal districts,
  Eircode routing keys and apartment flags from free-text addresses. Data-quality checks gate the pipeline.
- **Price model:** the target is `log(price ÷ national market index)`, where the index is the *previous* 3 months'
  median. This lets a tree model cope with market inflation without leaking future information. Evaluation is a
  strict time split (train on the past, test on the most recent 12 months) against two baselines. Explanations
  use exact TreeSHAP values from LightGBM.
- **Forecasting:** per county, three models are backtested on the last 12 months; the best is refit and used,
  with intervals sized from the backtest error.
- **LLM layer:** schema-grounded prompt with few-shot examples. Guardrails allow a single SELECT/WITH statement
  only, block file/network functions, run on a read-only connection with a row limit, and retry once with the
  error message on failure. Accuracy is measured on a gold-SQL benchmark by comparing results.

## Limitations and next steps

- The PPR has no bedrooms, bathrooms or floor area for most sales, so the model prices *location and type*.
- Next: geocode addresses (distance to city centre, DART/Luas), join CSO census small-area data and RTB rents,
  and add BER ratings.

## Data

Property Price Register © Property Services Regulatory Authority, used under its published terms. This project is
for research and education, and its outputs are not valuations or financial advice.
