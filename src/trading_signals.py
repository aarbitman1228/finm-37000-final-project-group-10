"""Generate and cache trading signals from Order Book Imbalance (OBI) data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INPUT_DIR = ROOT / "data" / "processed"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "signals"
DEFAULT_THRESHOLD = 0.75


def generate_signals(
    df: pd.DataFrame,
    threshold: float = DEFAULT_THRESHOLD,
) -> pd.DataFrame:
    """Generate buy and sell signals based on OBI thresholds."""
    if not 0 < threshold <= 1:
        raise ValueError("Threshold must be between 0 and 1.")

    if "OBI" not in df.columns:
        raise ValueError("Input DataFrame must contain an OBI column.")

    result = df.copy()

    result["signal"] = np.select(
        [
            result["OBI"] > threshold,
            result["OBI"] < -threshold,
        ],
        [1, -1],
        default=0,
    )

    return result


def process_signal_files(
    input_dir: Path,
    output_dir: Path,
    threshold: float = DEFAULT_THRESHOLD,
) -> list[Path]:
    """Generate and cache trading signals for all processed OBI Parquet files."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if not 0 < threshold <= 1:
        raise ValueError("Threshold must be between 0 and 1.")

    parquet_files = sorted(input_dir.glob("*_OBI_levels_*.parquet"))

    if not parquet_files:
        raise FileNotFoundError(f"No OBI Parquet files found in {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []

    for file in parquet_files:
        threshold_label = str(threshold).replace(".", "_")
        output_path = output_dir / f"{file.stem}_signals_{threshold_label}.parquet"

        if output_path.exists():
            print(f"Using cached signal data: {output_path}", flush=True)
            saved_paths.append(output_path)
            continue

        print(f"Processing: {file.name}", flush=True)

        df = pd.read_parquet(file)

        signals_df = generate_signals(df, threshold)

        signals_df.to_parquet(output_path, index=False)

        print(f"Saved signal data: {output_path}", flush=True)

        saved_paths.append(output_path)

        del df, signals_df

    return saved_paths


def main() -> None:
    """Process all OBI Parquet files using the default threshold."""
    process_signal_files(
        input_dir=DEFAULT_INPUT_DIR,
        output_dir=DEFAULT_OUTPUT_DIR,
        threshold=DEFAULT_THRESHOLD,
    )


if __name__ == "__main__":
    main()