import argparse
import os
from pathlib import Path

import duckdb
import snowflake.connector
from snowflake.connector.pandas_tools import write_pandas

ROOT = Path(__file__).resolve().parents[1]
DUCKDB_FILE = ROOT / "data" / "warehouse" / "banksight.duckdb"
CHUNK_ROWS = 50_000

# (duckdb schema, duckdb table, snowflake schema, snowflake table)
TABLES = [
    ("raw", "cfpb_complaints", "RAW", "CFPB_COMPLAINTS"),
    ("features", "complaint_labels", "FEATURES", "COMPLAINT_LABELS"),
]
TEXT_COLUMNS = ["consumer_complaint_narrative", "narrative", "input_text"]


def connect():
    """Open a Snowflake connection with key-pair login."""
    return snowflake.connector.connect(
        account=os.environ["SNOWFLAKE_ACCOUNT"],
        user=os.environ["SNOWFLAKE_USER"],
        private_key_file=os.environ["SNOWFLAKE_PRIVATE_KEY_PATH"],
        warehouse=os.environ.get("SNOWFLAKE_WAREHOUSE", "banksight_wh"),
        role=os.environ.get("SNOWFLAKE_ROLE", "banksight_role"),
        database="BANKSIGHT",
    )


def load_table(duck, conn, duck_schema, duck_table, sf_schema, sf_table, limit=None, drop_text=False):
    """Stream one DuckDB table into Snowflake in chunks. Returns the number of rows sent."""
    sql = f"select * from {duck_schema}.{duck_table}"
    if limit:
        sql += f" limit {int(limit)}"
    reader = duck.execute(sql).fetch_record_batch(CHUNK_ROWS)

    rows_sent = 0
    first = True
    for batch in reader:
        df = batch.to_pandas()
        if drop_text:
            df = df.drop(columns=[c for c in TEXT_COLUMNS if c in df.columns])
        df.columns = [c.upper() for c in df.columns]      # unquoted dbt SQL needs UPPERCASE names

        ok, n_chunks, n_rows, _ = write_pandas(
            conn, df, sf_table,
            database="BANKSIGHT", schema=sf_schema,
            auto_create_table=first,      # first chunk creates the table
            overwrite=first,              # ...and replaces any old copy, so re-runs are safe
            quote_identifiers=False,
        )
        if not ok:
            raise RuntimeError(f"write_pandas failed for {sf_schema}.{sf_table}")
        rows_sent += n_rows
        first = False
        print(f"  {sf_schema}.{sf_table}: {rows_sent:,} rows")
    return rows_sent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="only first N rows per table (for a test)")
    parser.add_argument("--drop-text", action="store_true", help="do not upload narrative text")
    args = parser.parse_args()

    duck = duckdb.connect(str(DUCKDB_FILE), read_only=True)
    conn = connect()
    try:
        for duck_schema, duck_table, sf_schema, sf_table in TABLES:
            print(f"Loading {duck_schema}.{duck_table} ...")
            n = load_table(duck, conn, duck_schema, duck_table, sf_schema, sf_table,
                           limit=args.limit, drop_text=args.drop_text)
            print(f"  done: {n:,} rows")
    finally:
        conn.close()
        duck.close()


if __name__ == "__main__":
    main()
