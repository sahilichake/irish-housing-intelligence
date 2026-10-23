"""Run the whole pipeline end to end.

    python -m src.pipeline            # real Property Price Register data
    python -m src.pipeline --sample   # synthetic data, for offline testing only
    python -m src.pipeline --refresh  # re-download the latest PPR file
"""
import argparse
import sys

from src.data import clean as clean_mod
from src.data.database import write_sales
from src.data.download import download_ppr
from src.models import forecast, train_price


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", action="store_true", help="use synthetic data (offline testing)")
    ap.add_argument("--refresh", action="store_true", help="force re-download of the PPR file")
    args = ap.parse_args()

    if args.sample:
        from src.data.sample import write_sample
        write_sample()
    elif not download_ppr(force=args.refresh):
        sys.exit(1)

    print("\n=== 1/4 Clean & validate ===")
    df = clean_mod.clean(clean_mod.read_raw())
    failures = clean_mod.validate(df)
    if failures:
        sys.exit(f"Data-quality checks failed: {failures}")
    print("[validate] all data-quality checks passed")

    print("\n=== 2/4 Load to DuckDB ===")
    write_sales(df, source="sample" if args.sample else "ppr")

    print("\n=== 3/4 Train price model ===")
    train_price.train()

    print("\n=== 4/4 Forecast county prices ===")
    forecast.run()

    print("\nDone. Start the app with:  streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    main()
