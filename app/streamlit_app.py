"""Irish Housing Market Intelligence — Streamlit app.

Run from the project root:   streamlit run app/streamlit_app.py
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from src.config import (COUNTIES, DB_PATH, FORECAST_METRICS_PATH, FORECAST_PATH,  # noqa: E402
                        LLM_EVAL_PATH, PRICE_METRICS_PATH, PRICE_MODEL_PATH)
from src.data.database import data_source, query  # noqa: E402

st.set_page_config(page_title="Irish Housing Intelligence", page_icon="🏠", layout="wide")

GREEN = "#169B62"
FEATURE_LABELS = {
    "county": "County", "locality": "Town / area", "dublin_district": "Dublin district", "eircode_routing": "Eircode area",
    "size_band": "Floor-area band", "is_new": "New build", "is_apartment": "Apartment",
    "has_eircode": "Has Eircode", "month": "Month of sale",
    "prev_rel_price": "Previous sale of this home", "years_since_prev": "Years since last sale",
    "n_prev_sales": "Times sold before", "street_rel_price": "Street price level", "street_n_sales": "Street sales history",
    "street_last": "Last sale on street", "street_recent": "Recent street prices", "street_days_since": "Time since last street sale",
    "estate_rel_price": "Estate price level", "estate_n_sales": "Estate sales history", "estate_recent": "Recent estate prices",
    "no_number": "Named house (no number)", "n_parts": "Address detail", "street_type": "Street type",
}


def eur(x: float) -> str:
    return f"€{x:,.0f}"


# ---------------------------------------------------------------- guards
if not DB_PATH.exists() or not PRICE_MODEL_PATH.exists():
    st.error("No data or model found. From the project root run:\n\n`python -m src.pipeline`")
    st.stop()


@st.cache_data(show_spinner=False)
def load_yearly() -> pd.DataFrame:
    return query("""SELECT county, year, COUNT(*) AS n_sales, MEDIAN(price_eur) AS median_price,
                    MEDIAN(price_eur) FILTER (WHERE is_new) AS median_new,
                    MEDIAN(price_eur) FILTER (WHERE NOT is_new) AS median_second_hand
                    FROM sales GROUP BY county, year ORDER BY year""")


@st.cache_data(show_spinner=False)
def load_national_monthly() -> pd.DataFrame:
    return query("SELECT period, COUNT(*) AS n_sales, MEDIAN(price_eur) AS median_price "
                 "FROM sales GROUP BY period ORDER BY period")


@st.cache_data(show_spinner=False)
def load_json(path: str) -> dict | None:
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else None


# ---------------------------------------------------------------- sidebar
st.sidebar.title("🏠 Irish Housing Intelligence")
page = st.sidebar.radio("Navigate", ["Market overview", "Price estimator", "Forecasts",
                                     "Ask the data (AI)", "Model card"])
source = data_source()
if source == "sample":
    st.sidebar.warning("Running on **synthetic sample data**. Run `python -m src.pipeline` for real PPR data.")
else:
    st.sidebar.caption("Data: Property Price Register (propertypriceregister.ie)")
st.sidebar.caption("Built by Sahil · [GitHub](https://github.com/)")

yearly = load_yearly()
latest_year = int(yearly["year"].max())

# ================================================================ OVERVIEW
if page == "Market overview":
    st.title("Irish residential property market")
    nat = load_national_monthly()
    y_now = query("SELECT MEDIAN(price_eur) m, COUNT(*) n FROM sales WHERE year = ?", [latest_year]).iloc[0]
    y_prev = query("SELECT MEDIAN(price_eur) m, COUNT(*) n FROM sales WHERE year = ?", [latest_year - 1]).iloc[0]
    total = query("SELECT COUNT(*) n FROM sales").iloc[0, 0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"Median price {latest_year} (YTD)", eur(y_now.m), f"{(y_now.m / y_prev.m - 1) * 100:+.1f}% vs {latest_year - 1}")
    c2.metric(f"Sales {latest_year} (YTD)", f"{int(y_now.n):,}")
    c3.metric("Sales recorded since 2010", f"{int(total):,}")
    c4.metric("Latest month in data", nat["period"].max().strftime("%b %Y"))

    st.subheader("National median price and volume")
    fig = go.Figure()
    fig.add_bar(x=nat["period"], y=nat["n_sales"], name="Sales per month", marker_color="#cfd8dc", yaxis="y2")
    fig.add_scatter(x=nat["period"], y=nat["median_price"].rolling(3, min_periods=1).median(),
                    name="Median price (3-mo)", line=dict(color=GREEN, width=3))
    fig.update_layout(yaxis=dict(title="Median price (€)", tickformat=",.0f"),
                      yaxis2=dict(title="Sales", overlaying="y", side="right", showgrid=False, tickformat=",.0f"),
                      legend=dict(orientation="h", y=1.1), height=420, margin=dict(t=30))
    st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.subheader("Compare counties over time")
        picks = st.multiselect("Counties", COUNTIES, default=["Dublin", "Cork", "Galway", "Limerick"])
        sub = yearly[yearly["county"].isin(picks)]
        fig = px.line(sub, x="year", y="median_price", color="county", markers=True,
                      labels={"median_price": "Median price (€)", "year": ""})
        fig.update_layout(height=420, yaxis_tickformat=",.0f", legend_title=None)
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.subheader("Median price by county")
        yr = st.select_slider("Year", options=sorted(yearly["year"].unique()), value=latest_year)
        sub = yearly[yearly["year"] == yr].sort_values("median_price")
        fig = px.bar(sub, x="median_price", y="county", orientation="h", color_discrete_sequence=[GREEN],
                     labels={"median_price": "Median price (€)", "county": ""}, hover_data=["n_sales"])
        fig.update_layout(height=600, xaxis_tickformat=",.0f")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Biggest movers: % change in median price")
    base_year = st.selectbox("Since", sorted(yearly["year"].unique())[:-1], index=0)
    a = yearly[yearly["year"] == base_year].set_index("county")["median_price"]
    b = yearly[yearly["year"] == latest_year].set_index("county")["median_price"]
    chg = ((b / a - 1) * 100).dropna().sort_values(ascending=False).rename("change_pct").reset_index()
    fig = px.bar(chg, x="county", y="change_pct", color="change_pct", color_continuous_scale="Greens",
                 labels={"change_pct": f"% change {base_year}→{latest_year}", "county": ""})
    fig.update_layout(height=380, coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Dublin postal districts")
    dub = query("""SELECT dublin_district AS district, COUNT(*) AS sales, ROUND(MEDIAN(price_eur),0) AS median_price
                   FROM sales WHERE county='Dublin' AND year >= ? AND dublin_district LIKE 'D%'
                   GROUP BY 1 HAVING COUNT(*) >= 10 ORDER BY median_price DESC""", [latest_year - 1])
    st.caption(f"Sales in {latest_year - 1}–{latest_year}, districts with ≥10 sales")
    st.dataframe(dub.style.format({"median_price": "€{:,.0f}", "sales": "{:,}"}), use_container_width=True, hide_index=True)

# ================================================================ ESTIMATOR
elif page == "Price estimator":
    from src.models.predict import load_bundle, make_row, predict

    st.title("What is this property worth today?")
    st.caption("LightGBM model trained on every recorded sale, valued at the current market level. "
               "The PPR has no bedrooms/size for most sales, so treat this as a location-and-type estimate.")
    bundle = load_bundle()
    cats = bundle["categories"]

    c1, c2, c3 = st.columns(3)
    county = c1.selectbox("County", COUNTIES, index=COUNTIES.index("Dublin"))
    district = None
    if county == "Dublin":
        districts = sorted([d for d in cats["dublin_district"] if d.startswith("D")], key=lambda d: (len(d), d))
        district = c1.selectbox("Dublin postal district", ["(not specified)"] + districts)
    towns = query("SELECT locality, COUNT(*) n FROM sales WHERE county = ? AND locality NOT IN ('OTHER','UNKNOWN') "
                  "GROUP BY 1 ORDER BY 1", [county])
    town = c1.selectbox("Town / area", ["(not specified)"] + [t.title() for t in towns["locality"]])
    ptype = c2.radio("Property type", ["House", "Apartment"], horizontal=True)
    condition = c2.radio("Condition", ["Second-hand", "New build"], horizontal=True)
    size = c3.selectbox("Floor area (if known)", ["unknown", "<38m2", "38-125m2", ">125m2"])
    eircode = c3.text_input("Eircode (optional)", placeholder="e.g. D04 X2Y3")
    address = st.text_input("Street address (optional, improves the estimate)",
                            placeholder="e.g. 10 Lismore Rd, Kimmage, Dublin 12",
                            help="Used to find earlier sales of this home and its street in the register.")
    submitted = st.button("Estimate price", type="primary")

    if submitted:
        area = None
        if county == "Dublin":
            area = district if district and district != "(not specified)" else "Co. Dublin (no district)"
        row = make_row(county, condition == "New build", ptype == "Apartment", size, eircode, area,
                       month=pd.Timestamp.today().month,
                       town="UNKNOWN" if town == "(not specified)" else town, address=address)
        town_q = None if town == "(not specified)" else town.upper()
        res = predict(row)
        p = res["predicted_price_eur"]
        lg = (load_json(str(PRICE_METRICS_PATH)) or {}).get("lightgbm", {})
        mape = lg.get("MdAPE_pct", lg.get("MAPE_pct", 20)) / 100

        c1, c2, c3 = st.columns(3)
        c1.metric("Estimated price", eur(p))
        c2.metric("Typical error range", f"{eur(round(p * (1 - mape), -3))} – {eur(round(p * (1 + mape), -3))}",
                  help="± the model's median % error on the test set (half of estimates are closer than this)")
        c3.metric("National market level", eur(res["market_level_eur"]), help="Lagged 3-month national median")

        if address.strip():
            prev = res.get("previous_sale")
            msg = (f"Found a previous sale of this home: {eur(prev['price_eur'])} on {prev['date']}. "
                   if prev else "No previous sale of this exact address found. ")
            msg += f"Used {res.get('street_sales_used', 0)} earlier sales on the same street."
            st.info(msg)

        st.subheader("Why this price? (SHAP explanation)")
        eff = pd.DataFrame([(FEATURE_LABELS.get(k, k), v) for k, v in res["feature_effects_pct"].items()],
                           columns=["feature", "effect_pct"]).sort_values("effect_pct")
        fig = px.bar(eff, x="effect_pct", y="feature", orientation="h",
                     color=eff["effect_pct"] > 0, color_discrete_map={True: GREEN, False: "#d9534f"},
                     labels={"effect_pct": "Effect on price vs. typical Irish sale (%)", "feature": ""})
        fig.update_layout(showlegend=False, height=380)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Comparable recent sales")
        comps = query("""SELECT sale_date, address, price_eur FROM sales
                         WHERE county = ? AND is_apartment = ? AND is_new = ?
                           AND sale_date >= (SELECT MAX(sale_date) FROM sales) - INTERVAL 12 MONTH
                           AND (? IS NULL OR dublin_district = ?)
                           AND (? IS NULL OR locality = ?)
                         ORDER BY sale_date DESC""",
                      [county, ptype == "Apartment", condition == "New build", area, area, town_q, town_q])
        if comps.empty:
            st.info("No comparable sales in the last 12 months.")
        else:
            st.write(f"{len(comps):,} similar sales in the last 12 months · median **{eur(comps.price_eur.median())}**")
            fig = px.histogram(comps, x="price_eur", nbins=40, color_discrete_sequence=["#90a4ae"],
                               labels={"price_eur": "Sale price (€)"})
            fig.add_vline(x=p, line_color=GREEN, line_width=3, annotation_text="Estimate")
            fig.update_layout(height=320, xaxis_tickformat=",.0f")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(comps.head(15).style.format({"price_eur": "€{:,.0f}"}), hide_index=True,
                         use_container_width=True)

# ================================================================ FORECASTS
elif page == "Forecasts":
    st.title("12-month price forecasts by county")
    fc = pd.read_csv(FORECAST_PATH, parse_dates=["period"])
    report = load_json(str(FORECAST_METRICS_PATH))
    options = ["Ireland (all)"] + [c for c in COUNTIES if c in report]
    county = st.selectbox("Area", options)

    sub = fc[fc["county"] == county]
    hist, fut = sub[sub["kind"] == "history"], sub[sub["kind"] == "forecast"]
    fig = go.Figure()
    fig.add_scatter(x=pd.concat([fut["period"], fut["period"][::-1]]),
                    y=pd.concat([fut["upper"], fut["lower"][::-1]]), fill="toself",
                    fillcolor="rgba(22,155,98,0.15)", line=dict(width=0), name="80% interval")
    fig.add_scatter(x=hist["period"], y=hist["value"], name="Median price (3-mo)", line=dict(color="#37474f"))
    fig.add_scatter(x=fut["period"], y=fut["value"], name="Forecast", line=dict(color=GREEN, width=3, dash="dash"))
    fig.update_layout(height=450, yaxis_tickformat=",.0f", yaxis_title="€", legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, use_container_width=True)

    r = report[county]
    c1, c2, c3 = st.columns(3)
    c1.metric("Forecast change, next 12 months", f"{r['forecast_change_12m_pct']:+.1f}%")
    c2.metric("Model selected", r["chosen"].replace("_", " "))
    c3.metric("Backtest error (MAPE)", f"{r['backtest_mape_pct'][r['chosen']]:.1f}%")
    with st.expander("Backtest: all models on the last 12 months of history"):
        st.dataframe(pd.DataFrame(r["backtest_mape_pct"], index=["MAPE %"]), use_container_width=True)

    st.subheader("Expected 12-month change across counties")
    allc = pd.DataFrame([{"county": k, "change": v["forecast_change_12m_pct"]} for k, v in report.items()
                         if k != "Ireland (all)"]).sort_values("change", ascending=False)
    fig = px.bar(allc, x="county", y="change", color_discrete_sequence=[GREEN],
                 labels={"change": "Forecast % change", "county": ""})
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Forecasts are statistical extrapolations of past prices, not financial advice.")

# ================================================================ ASK THE DATA
elif page == "Ask the data (AI)":
    from src.llm.text_to_sql import UnsafeSQLError, ask

    st.title("Ask the data")
    st.caption("An LLM turns your question into SQL, a guardrail checks it is read-only, DuckDB runs it, "
               "and the LLM summarises the result. You can always see the SQL.")
    providers = ["ollama", "anthropic", "openai"]
    default = os.getenv("LLM_PROVIDER", "ollama").lower()
    provider = st.selectbox("LLM provider", providers, index=providers.index(default) if default in providers else 0,
                            help="Set the API key in .env. Ollama runs free on your own machine.")
    examples = ["How have median prices in Galway changed each year since 2015 compared with Cork?",
                "Which 5 Dublin districts were most expensive last year?",
                "What share of sales were new builds in each year?",
                "Which county had the biggest price increase between 2019 and 2024?"]
    cols = st.columns(len(examples))
    for col, ex in zip(cols, examples):
        if col.button(ex, use_container_width=True):
            st.session_state["q"] = ex
    q = st.text_input("Your question", key="q", placeholder="e.g. Median apartment price in Cork in 2023?")

    if st.button("Ask", type="primary") and q:
        with st.spinner("Thinking..."):
            try:
                res = ask(q, provider)
            except UnsafeSQLError as e:
                st.error(f"Blocked by the SQL guardrail: {e}")
                st.stop()
            except Exception as e:  # noqa: BLE001
                st.error(f"Could not answer: {e}")
                st.stop()
        if res.get("warning"):
            st.warning(f"This answer may be wrong: {res['warning']} Check the SQL below, or try rephrasing.")
        else:
            st.success(res["answer"])
        df = res["result"]
        num = df.select_dtypes("number").columns
        if len(df) > 1 and len(df.columns) >= 2 and len(num) >= 1:
            x = df.columns[0]
            y = [c for c in num if c != x]
            if y:
                kind = px.line if pd.api.types.is_numeric_dtype(df[x]) and df[x].nunique() > 5 else px.bar
                color = df.columns[1] if len(df.columns) > 2 and df.columns[1] not in num else None
                fig = kind(df, x=x, y=y[0], color=color,
                           color_discrete_sequence=None if color else [GREEN])
                st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df, use_container_width=True, hide_index=True)
        with st.expander("SQL used"):
            st.code(res["sql"], language="sql")

    ev = load_json(str(LLM_EVAL_PATH))
    if ev:
        st.caption(f"Evaluated on {ev['n']} benchmark questions: **{ev['execution_accuracy_pct']}%** execution accuracy.")

# ================================================================ MODEL CARD
elif page == "Model card":
    st.title("Model card")
    m = load_json(str(PRICE_METRICS_PATH))
    st.subheader("Price model — test on the most recent 12 months (never seen in training)")
    st.caption(f"Test period: {m['test_period']} · train {m['n_train']:,} rows · test {m['n_test']:,} rows")
    tbl = pd.DataFrame({"LightGBM": m["lightgbm"], "Baseline: county × market index": m["baseline_county_x_index"],
                        "Baseline: county median": m["baseline_county_median"]}).T
    tbl.columns = ["MAE (€)", "Median abs. error (€)", "MAPE (%)", "Median APE (%)", "Within 10% (%)", "R² (log)"]
    st.dataframe(tbl.style.format({"MAE (€)": "€{:,.0f}", "Median abs. error (€)": "€{:,.0f}",
                                   "MAPE (%)": "{:.1f}", "Median APE (%)": "{:.1f}", "Within 10% (%)": "{:.1f}", "R² (log)": "{:.3f}"}),
                 use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Global feature importance (mean |SHAP|)")
        imp = pd.DataFrame([(FEATURE_LABELS.get(k, k), v) for k, v in m["shap_importance"].items()],
                           columns=["feature", "importance"]).sort_values("importance")
        st.plotly_chart(px.bar(imp, x="importance", y="feature", orientation="h",
                               color_discrete_sequence=[GREEN]).update_layout(height=380), use_container_width=True)
    with c2:
        st.subheader("Error by county (MAE €)")
        bc = pd.Series(m["mae_by_county"]).sort_values().reset_index()
        bc.columns = ["county", "mae"]
        st.plotly_chart(px.bar(bc, x="mae", y="county", orientation="h", color_discrete_sequence=["#90a4ae"])
                        .update_layout(height=600, xaxis_tickformat=",.0f"), use_container_width=True)

    ev = load_json(str(LLM_EVAL_PATH))
    st.subheader("Text-to-SQL evaluation")
    if ev:
        st.metric("Execution accuracy", f"{ev['execution_accuracy_pct']}%", f"{ev['correct']}/{ev['n']} questions")
        st.dataframe(pd.DataFrame(ev["details"])[["question", "correct", "sql"]], use_container_width=True, hide_index=True)
    else:
        st.info("Run `python -m src.llm.evaluate` to benchmark the LLM layer.")

    st.subheader("Method and limitations")
    st.markdown("""
- **Target:** log(price ÷ lagged national market index). Lets the tree model handle market-wide inflation
  without leaking future prices (index uses only the previous 3 months).
- **Validation:** strict time split — train on the past, test on the most recent 12 months.
- **Sale history:** previous sale of the same home, plus the price level of earlier sales on the same street and
  estate (overall, most recent, and last 5 sale days). All are computed only from sales on *earlier days* than the
  one being valued, so same-day bulk deals never see each other. Address text adds clues such as named houses.
- **Training:** L1 loss (targets the typical error) with higher weight on recent years.
- **Data limits:** the PPR has no bedrooms, bathrooms or floor area for most sales, so the model prices
  *location and type*. Apartment status is inferred from the address text.
- **Excluded:** sales flagged *Not Full Market Price*, and prices outside €30k–€5m. New-build prices reported
  ex-VAT have 13.5% VAT added back.
- **Next steps:** geocode addresses for distance-to-amenity features; join CSO census small-area statistics;
  add BER ratings from listings data.
""")
