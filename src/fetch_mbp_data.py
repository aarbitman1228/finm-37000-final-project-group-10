"""Fetch and cache daily chunks of Databento MBP-10 futures data."""

from __future__ import annotations

import argparse
import os
from datetime import date, timedelta
from pathlib import Path

import databento as db
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
DATASET = "GLBX.MDP3"
SCHEMA = "mbp-10"
MAX_STREAM_BYTES = 5_000_000_000
DEFAULT_START = date(2026, 10, 4)
DEFAULT_END = date(2026, 10, 10)
DEFAULT_SYMBOL = "CL.c.0"


def parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Date must use YYYY-MM-DD format") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--symbol",
        default=DEFAULT_SYMBOL,
        choices=("CL.c.0", "GC.c.0", "SI.c.0"),
        help="Continuous front-month symbol (default: %(default)s)",
    )
    parser.add_argument(
        "--start",
        type=parse_date,
        default=DEFAULT_START,
        help="Inclusive start date; requests are split by day (default: %(default)s)",
    )
    parser.add_argument(
        "--end",
        type=parse_date,
        default=DEFAULT_END,
        help="Exclusive end date (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data" / "raw",
        help="Directory for cached Parquet data (default: data/raw)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Fetch again and replace matching daily files",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Confirm the estimated data cost without prompting",
    )
    return parser.parse_args()


def fetch_mbp_data(
    symbol: str,
    start: date,
    end: date,
    output_dir: Path,
    *,
    force: bool = False,
    confirm_cost: bool = False,
) -> list[Path] | None:
    if end <= start:
        raise ValueError("End date must be later than start date")

    output_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    day_start = start
    while day_start < end:
        day_end = min(day_start + timedelta(days=1), end)
        output_path = (
            output_dir
            / f"{symbol.replace('.', '_')}_{SCHEMA}_{day_start}_{day_end}.parquet"
        )
        dbn_path = output_path.with_suffix(".dbn.zst")

        if output_path.exists() and not force:
            print(f"Using cached data: {output_path}", flush=True)
        else:
            if dbn_path.exists() and not force:
                print(f"Found downloaded daily data: {dbn_path}", flush=True)
                needs_download = False
            else:
                needs_download = True
            jobs.append(
                {
                    "start": day_start,
                    "end": day_end,
                    "output_path": output_path,
                    "dbn_path": dbn_path,
                    "needs_download": needs_download,
                }
            )
        day_start = day_end

    if not jobs:
        print("All daily files are already cached.", flush=True)
        return []

    download_jobs = [job for job in jobs if job["needs_download"]]
    client = None
    if download_jobs:
        load_dotenv(ROOT / ".env")
        if not os.getenv("DATABENTO_API_KEY"):
            raise RuntimeError(
                "Databento API key not found. Add DATABENTO_API_KEY to the project .env file."
            )
        client = db.Historical()

        for job in download_jobs:
            request = {
                "dataset": DATASET,
                "symbols": [symbol],
                "schema": SCHEMA,
                "stype_in": "continuous",
                "start": job["start"].isoformat(),
                "end": job["end"].isoformat(),
            }
            billable_size = client.metadata.get_billable_size(**request)
            if billable_size > MAX_STREAM_BYTES:
                raise RuntimeError(
                    f"The daily request for {job['start']} is estimated at "
                    f"{billable_size:,} bytes, over the 5 GB streaming limit. "
                    "Split this date into smaller time ranges before downloading."
                )
            if billable_size == 0:
                print(f"No billable records reported for {job['start']}; skipping.", flush=True)
                job["skip"] = True
                continue
            job["request"] = request
            job["billable_size"] = billable_size
            job["estimated_cost"] = client.metadata.get_cost(**request)

        request_jobs = [job for job in download_jobs if not job.get("skip")]
        total_cost = sum(job["estimated_cost"] for job in request_jobs)
        for job in request_jobs:
            size_gb = job["billable_size"] / 1_000_000_000
            print(
                f"{job['start']}: {size_gb:.2f} GB, "
                f"estimated ${job['estimated_cost']:.4f}",
                flush=True,
            )
        print(f"Estimated total Databento data cost: ${total_cost:.4f}", flush=True)

        if request_jobs and not confirm_cost:
            answer = input("Download this data? [y/N] ").strip().lower()
            if answer != "y":
                print("Download cancelled; no market data was requested.", flush=True)
                return None

    saved_paths = []
    for job in jobs:
        if job.get("skip"):
            continue

        output_path = job["output_path"]
        dbn_path = job["dbn_path"]
        if job["needs_download"]:
            partial_path = output_path.with_suffix(".partial.dbn.zst")
            partial_path.unlink(missing_ok=True)
            print(
                f"Downloading {job['start']} through {job['end']} (end exclusive) "
                f"to {partial_path}.",
                flush=True,
            )
            assert client is not None
            try:
                data = client.timeseries.get_range(
                    **job["request"],
                    path=partial_path,
                )
            except db.BentoClientError as exc:
                partial_path.unlink(missing_ok=True)
                if exc.http_status != 402:
                    raise
                raise SystemExit(
                    "Databento rejected a daily request because the account budget is "
                    "insufficient. The daily Parquet files already completed were kept. "
                    "Increase the budget or add funds at "
                    "https://databento.com/portal/billing, then rerun the script."
                ) from None
            partial_path.replace(dbn_path)
            print(f"Download complete for {job['start']}; converting to Parquet.", flush=True)
        else:
            print(f"Converting cached DBN data for {job['start']}.", flush=True)
            data = db.DBNStore.from_file(dbn_path)

        data.to_parquet(output_path, compression="zstd")
        dbn_path.unlink(missing_ok=True)
        print(f"Saved daily Parquet file: {output_path}", flush=True)
        saved_paths.append(output_path)

    return saved_paths


def main() -> None:
    args = parse_args()
    fetch_mbp_data(
        symbol=args.symbol,
        start=args.start,
        end=args.end,
        output_dir=args.output_dir,
        force=args.force,
        confirm_cost=args.yes,
    )


if __name__ == "__main__":
    main()