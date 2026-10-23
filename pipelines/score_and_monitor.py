"""Scheduled pipeline: score a month of complaints with the champion model and check for drift. The reference and the windows
must use the same model, so build the reference again if you change it.
"""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PREFECT_SERVER_ANALYTICS_ENABLED", "false")   # no usage tracking, and fewer start-up messages

import duckdb
import numpy as np
import pandas as pd
from prefect import flow, task

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))   # so `import banksight` works without installing it

from banksight import config
from banksight.ml.loader import load_models
from banksight.ml.scoring import score_complaint
from banksight.monitoring.drift import add_distribution_parts, advice_for, compare_to_reference, overall_status, summarise_window
from banksight.nlp.text_cleaning import mask_for_b4

QUERY = """
    SELECT l.complaint_id, l.date_received, l.product_group, l.got_relief, r.issue, r.narrative
    FROM features.complaint_labels l
    JOIN raw.cfpb_complaints r USING (complaint_id)
    WHERE l.target_eligible
      AND r.narrative IS NOT NULL
      AND length(trim(r.narrative)) >= {min_chars}
      AND length(r.narrative) <= {max_chars}
      AND {where}
"""


@task(retries=2, retry_delay_seconds=5)
def load_complaints(where):
    """Read the complaints of one window from the warehouse (read-only, so it never blocks a notebook)."""
    connection = duckdb.connect(str(config.ROOT / "data" / "warehouse" / "banksight.duckdb"), read_only=True)
    complaints = connection.sql(QUERY.format(min_chars=config.MIN_NARRATIVE_CHARS, max_chars=config.MAX_NARRATIVE_CHARS, where=where)).df()
    connection.close()
    complaints["issue"] = complaints["issue"].fillna("Unknown").astype(str)
    complaints["got_relief"] = complaints["got_relief"].astype(float)
    return complaints


@task
def add_text_features(complaints, models):
    """Add masked text, word count and topic to every complaint (this is cheap, so it is done for all of them)."""
    complaints = complaints.copy()
    complaints["text"] = [mask_for_b4(narrative) for narrative in complaints["narrative"]]
    complaints["words"] = [len(narrative.split()) for narrative in complaints["narrative"]]

    topic_model = models["topics"]
    weights = topic_model["nmf"].transform(topic_model["vectorizer"].transform(complaints["text"]))
    complaints["topic_id"] = np.where(weights.sum(axis=1) > 0, weights.argmax(axis=1), -1)
    return complaints


@task
def score_sample(complaints, models, rows):
    """Score a random sample of the window with the champion (the model is slow, so we do not score everything)."""
    sample = complaints.sample(n=min(rows, len(complaints)), random_state=42)

    probabilities = []
    flags = []
    models_used = []
    for _, complaint in sample.iterrows():
        answer = score_complaint(models, complaint["product_group"], complaint["issue"], complaint["narrative"])
        probabilities.append(answer["relief_probability"])
        flags.append(answer["review_priority"] == "review_first")
        models_used.append(answer["model_used"])

    scored = pd.DataFrame({"probability": probabilities, "flagged": flags, "model_used": models_used})
    return scored


@flow(name="build-drift-reference")
def build_reference(rows, light):
    """Measure the validation period once. Later windows are compared with this picture."""
    models = load_models(use_roberta=not light)
    complaints = load_complaints("l.split = 'validation'")
    complaints = add_text_features(complaints, models)
    scored = score_sample(complaints, models, rows)

    snapshot = summarise_window(complaints, scored)
    snapshot = add_distribution_parts(snapshot, complaints, scored)
    snapshot["model"] = scored["model_used"].mode()[0]
    snapshot["window"] = "validation period"
    snapshot["topic_names"] = {str(topic): ("unclassified" if topic == -1 else str(models["topics"]["topic_labels"][topic]))
                               for topic in snapshot["topic_categories"]}

    config.DRIFT_DIR.mkdir(parents=True, exist_ok=True)
    config.REFERENCE_FILE.write_text(json.dumps(snapshot, indent=2))
    print("reference saved:", config.REFERENCE_FILE, "| model:", snapshot["model"], "| complaints:", snapshot["n_complaints"])
    return snapshot


@flow(name="monitor-month")
def monitor_months(months, rows, light):
    """Score each month and compare it with the reference. Saves one report per month and a running history."""
    if not config.REFERENCE_FILE.exists():
        raise SystemExit("No reference yet. Run: python pipelines/score_and_monitor.py build-reference")
    reference = json.loads(config.REFERENCE_FILE.read_text())
    if "known_template_keys" not in reference:
        raise SystemExit("The reference file is from an older version. Run build-reference again.")

    models = load_models(use_roberta=not light)
    history = []

    for month in months:
        start = pd.Timestamp(month + "-01")
        end = start + pd.offsets.MonthBegin(1)
        complaints = load_complaints(f"l.date_received >= DATE '{start.date()}' AND l.date_received < DATE '{end.date()}'")
        if len(complaints) == 0:
            print(month, ": no complaints, skipped")
            continue

        complaints = add_text_features(complaints, models)
        scored = score_sample(complaints, models, rows)
        if scored["model_used"].mode()[0] != reference["model"]:
            raise SystemExit(f"The reference was built with '{reference['model']}' but this run used '{scored['model_used'].mode()[0]}'. "
                             "Build the reference again with the same model.")

        snapshot = summarise_window(complaints, scored, reference)
        snapshot = add_distribution_parts(snapshot, complaints, scored, reference)
        report = compare_to_reference(reference, snapshot, month)
        report["n_complaints"] = snapshot["n_complaints"]

        report.to_csv(config.DRIFT_DIR / f"drift_{month}.csv", index=False)
        history.append(report)

        print(f"\n=== {month}: {snapshot['n_complaints']} complaints, {snapshot['n_scored']} scored | overall: {overall_status(report).upper()}")
        print(report[["measure", "reference", "current", "change", "status"]].round(4).to_string(index=False))
        for line in advice_for(report):
            print("  ->", line)

    if len(history) > 0:
        new_rows = pd.concat(history, ignore_index=True)
        if config.HISTORY_FILE.exists():
            old_rows = pd.read_csv(config.HISTORY_FILE)
            old_rows = old_rows[~old_rows["window"].isin(new_rows["window"])]    # re-running a month replaces it
            new_rows = pd.concat([old_rows, new_rows], ignore_index=True)
        new_rows.to_csv(config.HISTORY_FILE, index=False)
        print("\nhistory saved:", config.HISTORY_FILE)


def test_period_months():
    """Every month that has complaints after the validation period."""
    connection = duckdb.connect(str(config.ROOT / "data" / "warehouse" / "banksight.duckdb"), read_only=True)
    dates = connection.sql("SELECT DISTINCT strftime(date_received, '%Y-%m') AS month FROM features.complaint_labels "
                           "WHERE split = 'test' ORDER BY month").df()
    connection.close()
    return dates["month"].tolist()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BankSight scoring and drift pipeline")
    parser.add_argument("command", choices=["build-reference", "monitor", "serve"])
    parser.add_argument("--month", help="one month to check, like 2025-11")
    parser.add_argument("--all", action="store_true", help="check every month of the test period")
    parser.add_argument("--rows", type=int, default=config.SCORED_ROWS_PER_WINDOW, help="complaints to score per window")
    parser.add_argument("--light", action="store_true", help="use the fast B4 model instead of RoBERTa")
    arguments = parser.parse_args()

    if arguments.command == "build-reference":
        build_reference(arguments.rows, arguments.light)
    elif arguments.command == "monitor":
        months = test_period_months() if arguments.all else [arguments.month]
        if months == [None]:
            raise SystemExit("Give --month 2025-11 or --all")
        monitor_months(months, arguments.rows, arguments.light)
    elif arguments.command == "serve":
        # runs the monitor on the 2nd of every month at 06:00 for the previous month. Needs this process to stay running.
        # (Written but not tested in this project.)
        previous_month = (pd.Timestamp.today() - pd.offsets.MonthBegin(1)).strftime("%Y-%m")
        monitor_months.serve(name="banksight-monthly-monitor", cron="0 6 2 * *",
                             parameters={"months": [previous_month], "rows": arguments.rows, "light": arguments.light})
