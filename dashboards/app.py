"""BankSight dashboard. Start it from the project folder with:   python -m streamlit run dashboards/app.py

It only READS the files the notebooks and the pipeline already saved (reports/...), so it is fast and cannot change anything.
The last tab can run the model directly (no API needed) or through the API.
"""
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
sys.path.insert(0, str(ROOT / "src"))     # so `import banksight` works without installing it
MODELS_PRESENT = (ROOT / "models" / "04_champion_bundle.joblib").exists()   # False on the online demo (models are not in git)

st.set_page_config(page_title="BankSight", layout="wide")
st.title("BankSight: consumer complaint triage")
st.caption("Ranks complaints by how likely they are to end with relief, so reviewers can look at the likeliest ones first. "
           "A triage aid only: it never decides that a customer gets nothing.")
if not MODELS_PRESENT:
    st.info("Online demo mode: the dashboard shows saved reports only. Model files are too large for GitHub, "
            "so live scoring works only when you run the project on your own computer.")


@st.cache_data
def load_csv(relative_path):
    """Read one saved report. Returns None if the notebook that makes it has not been run yet."""
    path = REPORTS / relative_path
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_resource
def get_models():
    """Load the models once and keep them in memory (only when someone clicks 'Score it' for the first time)."""
    from banksight.ml.loader import load_models
    return load_models()


def score_complaint(models, product_group, issue, narrative):
    """Score one complaint in this app (same function the API uses)."""
    from banksight.ml.scoring import score_complaint as score
    return score(models, product_group, issue, narrative)


def say_missing(file_name, made_by):
    st.info(f"`{file_name}` is not there yet. It is created by {made_by}.")


tab_overview, tab_products, tab_model, tab_drift, tab_score = st.tabs(
    ["Overview", "Products", "Model results", "Drift monitor", "Score a complaint"])

# ---------------------------------------------------------------- Overview
with tab_overview:
    monthly = load_csv("kpis/monthly_kpis.csv")
    headline = load_csv("test/09_headline.csv")

    if monthly is None:
        say_missing("reports/kpis/monthly_kpis.csv", "notebook 06")
    else:
        eligible = monthly["eligible_complaints"].sum()
        overall_rate = monthly["relief_cases"].sum() / eligible
        first, second, third, fourth = st.columns(4)
        first.metric("Complaints", f"{int(monthly['complaints'].sum()):,}")
        second.metric("With a final outcome", f"{int(eligible):,}")
        third.metric("Share that got relief", f"{overall_rate:.1%}")
        if headline is not None:
            champion_row = headline[headline["model"].str.contains("RoBERTa")].iloc[0]
            fourth.metric("Test PR-AUC (RoBERTa)", f"{champion_row['pr_auc']:.3f}")

        st.subheader("Share of complaints that end with relief, by month")
        chart_data = monthly.assign(month=pd.to_datetime(monthly["month"])).set_index("month")
        st.line_chart(chart_data[["relief_rate", "relief_rate_3m"]])
        st.caption("`relief_rate_3m` is the 3-month average. Recent months are the least settled, because outcomes arrive late.")

        st.subheader("Complaints per month")
        st.bar_chart(chart_data["complaints"])

# ---------------------------------------------------------------- Products
with tab_products:
    products = load_csv("kpis/product_kpis.csv")
    issues = load_csv("kpis/top_issues_per_product.csv")

    if products is None:
        say_missing("reports/kpis/product_kpis.csv", "notebook 06")
    else:
        st.subheader("Relief rate by product group")
        st.bar_chart(products.set_index("product_group")["relief_rate"])
        st.dataframe(products, hide_index=True)

    if issues is not None:
        st.subheader("Top issues inside one product")
        chosen_product = st.selectbox("Product group", sorted(issues["product_group"].unique()), key="issues_product")
        st.dataframe(issues[issues["product_group"] == chosen_product], hide_index=True)

# ---------------------------------------------------------------- Model results
with tab_model:
    st.subheader("Sealed test result (opened once)")
    test_headline = load_csv("test/09_headline.csv")
    test_bootstrap = load_csv("test/09_bootstrap.csv")

    if test_headline is None:
        say_missing("reports/test/09_headline.csv", "notebook 09")
    else:
        st.dataframe(test_headline.round(4), hide_index=True)
        if test_bootstrap is not None:
            row = test_bootstrap.iloc[0]
            st.success(f"RoBERTa minus B4 on PR-AUC: {row['gain']:+.4f} (95% interval {row['ci_low']:+.4f} to {row['ci_high']:+.4f}). {row['verdict']}")

    left, right = st.columns(2)
    with left:
        st.markdown("**By slice** (templates cannot help in the last two)")
        by_subset = load_csv("test/09_by_subset.csv")
        if by_subset is not None:
            st.dataframe(by_subset.pivot(index="model", columns="subset", values="pr_auc").round(4))
    with right:
        st.markdown("**Are the probabilities honest?** (lower ECE is better)")
        calibration = load_csv("test/09_calibration.csv")
        if calibration is not None:
            st.dataframe(calibration.round(4), hide_index=True)

    st.markdown("**The 25% decision rule on the test period**")
    decision = load_csv("test/09_decision_rule.csv")
    if decision is not None:
        st.dataframe(decision.round(4), hide_index=True)

    st.markdown("**Reviewing only the top k%: share of relief cases caught**")
    lift = load_csv("test/09_lift.csv")
    if lift is not None:
        st.line_chart(lift.pivot(index="reviewed_top_share", columns="model", values="relief_cases_caught"))

    st.subheader("How the champion was chosen (validation)")
    validation = load_csv("transformer/08c_overall_comparison.csv")
    verdict = load_csv("transformer/08c_bootstrap_verdict.csv")
    if validation is not None:
        st.dataframe(validation.round(4), hide_index=True)
    if verdict is not None:
        st.dataframe(verdict.round(4), hide_index=True)
    st.caption("Rule fixed in advance: a challenger must beat B4 by at least 0.01 PR-AUC, with a bootstrap interval above 0 "
               "(widened for the number of comparisons).")

# ---------------------------------------------------------------- Drift monitor
with tab_drift:
    st.subheader("Is recent data still like the data the model was checked on?")
    history = load_csv("drift/drift_history.csv")

    if history is None:
        say_missing("reports/drift/drift_history.csv", "`python pipelines/score_and_monitor.py monitor --all`")
    else:
        if "details" not in history.columns:
            history["details"] = None
        st.warning("An alert means the DATA changed, not that the model is wrong. Check what moved first: "
                   "often a data-source change explains many alerts at once.")

        icons = {"ok": "🟢 ok", "warn": "🟡 warn", "alert": "🔴 alert"}
        history["status_label"] = history["status"].map(icons).fillna(history["status"])
        status_table = history.pivot_table(index="window", columns="measure", values="status_label", aggfunc="last")
        st.dataframe(status_table)

        st.subheader("Population Stability Index over time")
        psi_table = history[history["measure"].isin(["topic_psi", "product_psi", "score_psi"])].pivot_table(index="window", columns="measure", values="change", aggfunc="last")
        st.line_chart(psi_table)
        st.caption("Under 0.10 stable, 0.10 to 0.25 some change, over 0.25 a large change.")

        st.subheader("What moved most in a given month")
        chosen_window = st.selectbox("Month", sorted(history["window"].unique(), reverse=True), key="drift_month")
        window_rows = history[(history["window"] == chosen_window) & (history["details"].notna())]
        for _, row in window_rows.iterrows():
            st.write(f"**{row['measure']}** ({row['status']}): {row['details']}")

# ---------------------------------------------------------------- Score a complaint
with tab_score:
    st.subheader("Paste a complaint and get its triage score")
    st.caption("The model reads the complaint text, the product and the issue, and returns a calibrated chance of relief. "
               "It helps decide which complaints a person should look at first. It never decides to deny anything.")

    if not MODELS_PRESENT:
        st.warning("Live scoring is switched off here because the model files are not in this deployment. "
                   "Clone the repo, add the files to models/, and run it locally to try it.")
        st.stop()      # this is the last tab, so stopping here hides nothing else

    mode = st.radio("Run the model", ["Directly in this app", "Through the API"], horizontal=True, key="score_mode")
    if mode == "Through the API":
        api_url = st.text_input("API address", "http://127.0.0.1:8000", key="score_api_url")
        st.caption("Start it first:  python -m uvicorn banksight.api.main:app --app-dir src")

    product_table = load_csv("kpis/product_kpis.csv")
    if product_table is not None:
        product_group = st.selectbox("Product group", sorted(product_table["product_group"].unique()), key="score_product")
    else:
        product_group = st.text_input("Product group", "Credit or prepaid card", key="score_product_text")

    issue = st.text_input("Issue (optional)", "", key="score_issue")
    narrative = st.text_area("Complaint text (at least 20 characters)", height=180, key="score_text")

    if st.button("Score it", key="score_button"):
        answer = None

        if mode == "Directly in this app":
            try:
                with st.spinner("Loading the models. The first time takes about 30 seconds."):
                    models = get_models()
            except Exception as error:          # missing file, missing package, etc.
                st.error(f"Could not load the models: {error}")
                st.stop()
            if product_group not in models["b4"]["product_cats"]:
                st.error(f"The model does not know the product group '{product_group}'.")
            else:
                try:
                    answer = score_complaint(models, product_group, issue or None, narrative)
                except ValueError as error:
                    st.error(str(error))
        else:
            request = urllib.request.Request(
                api_url + "/predict",
                data=json.dumps({"product_group": product_group, "issue": issue or None, "narrative": narrative}).encode(),
                headers={"Content-Type": "application/json"})
            try:
                answer = json.loads(urllib.request.urlopen(request, timeout=60).read())
            except urllib.error.HTTPError as error:
                st.error(f"The API said: {error.read().decode()[:300]}")
            except (urllib.error.URLError, OSError):
                st.error("Could not reach the API. Is it running?")

        if answer is not None:
            left, middle, right = st.columns(3)
            left.metric("Chance of relief", f"{answer['relief_probability']:.1%}")
            middle.metric("Review priority", answer["review_priority"].replace("_", " "))
            right.metric("Model", answer["model_used"])
            st.write("Topic:", answer["topic_label"])
            st.caption(f"'review first' means the chance is at or above {answer['threshold']:.0%}, the break-even point we set from the cost of a wasted review against the value of a caught relief case.")
            st.info(answer["notice"])