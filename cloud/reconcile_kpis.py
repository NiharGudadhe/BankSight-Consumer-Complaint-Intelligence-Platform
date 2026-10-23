import sys
from pathlib import Path

import duckdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from load_to_snowflake import DUCKDB_FILE, ROOT, connect  # noqa: E402

TOLERANCE = 1e-6


def main():
    duck = duckdb.connect(str(DUCKDB_FILE), read_only=True)
    left = duck.execute("select * from analytics.mart_monthly_kpis").df()
    duck.close()

    conn = connect()
    cur = conn.cursor()
    cur.execute("select * from BANKSIGHT.ANALYTICS.MART_MONTHLY_KPIS")
    right = cur.fetch_pandas_all()
    conn.close()

    left.columns = [c.lower() for c in left.columns]
    right.columns = [c.lower() for c in right.columns]
    key = left.columns[0]                       # first column is the month
    left[key] = pd.to_datetime(left[key]).dt.strftime("%Y-%m")
    right[key] = pd.to_datetime(right[key]).dt.strftime("%Y-%m")

    merged = left.merge(right, on=key, how="outer", suffixes=("_duck", "_snow"), indicator=True)
    problems = int((merged["_merge"] != "both").sum())

    rows = []
    for col in left.columns:
        if col == key or not pd.api.types.is_numeric_dtype(left[col]):
            continue
        diff = (merged[col + "_duck"].astype(float) - merged[col + "_snow"].astype(float)).abs()
        rows.append({"column": col, "max_abs_diff": diff.max(), "months_off": int((diff > TOLERANCE).sum())})
        problems += int((diff > TOLERANCE).sum())

    report = pd.DataFrame(rows)
    out = ROOT / "reports" / "cloud"
    out.mkdir(parents=True, exist_ok=True)
    report.to_csv(out / "reconcile_monthly_kpis.csv", index=False)
    print(report.to_string(index=False))
    print("\nMATCH" if problems == 0 else f"\nMISMATCH: {problems} problem(s)")
    sys.exit(0 if problems == 0 else 1)


if __name__ == "__main__":
    main()
