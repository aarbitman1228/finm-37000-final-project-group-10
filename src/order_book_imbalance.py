"""Calculate and cache weighted Order Book Imbalance (OBI) from Databento MBP-10 data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INPUT_DIR = ROOT / "data" / "raw"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "processed"
DEFAULT_LEVELS = [0]


def get_weights(list_level: list[int]) -> dict[str, float]:
    """Calculate normalized linear weights for selected order book levels."""
    list_str_level = [f"{level:02d}" for level in list_level]
    list_weights = [10 - x for x in list_level]
    total_weight = sum(list_weights)
    list_weights = [x / total_weight for x in list_weights]
    dict_weights = dict(zip(list_str_level, list_weights))
    return dict_weights


def get_weighted_bid_ask(
    df_full: pd.DataFrame,
    list_level: list[int],
) -> pd.DataFrame:
    """Calculate weighted bid and ask sizes across selected order book levels."""
    if not list_level or any(level < 0 or level > 9 for level in list_level):
        raise ValueError(
            "All levels must be between 0 and 9, and the list cannot be empty."
        )

    if len(list_level) != len(set(list_level)):
        raise ValueError("Levels must be unique.")

    dict_level = get_weights(list_level)

    bid_cols = ["bid_sz_" + liq_level for liq_level in dict_level]
    ask_cols = ["ask_sz_" + liq_level for liq_level in dict_level]

    df_bid = df_full[bid_cols].copy().astype(float)
    df_ask = df_full[ask_cols].copy().astype(float)

    avg_bid = np.dot(df_bid[bid_cols], list(dict_level.values()))
    avg_ask = np.dot(df_ask[ask_cols], list(dict_level.values()))

    df_bidask = df_full[["ts_event"]].copy()
    df_bidask["bid"] = avg_bid
    df_bidask["ask"] = avg_ask

    return df_bidask


def get_OBI(
    info_df: pd.DataFrame,
    list_levels: list[int],
) -> pd.DataFrame:
    """Calculate Order Book Imbalance using weighted bid and ask liquidity."""
    df_bid_ask = get_weighted_bid_ask(info_df, list_levels)

    denominator = df_bid_ask["bid"] + df_bid_ask["ask"]

    df_bid_ask["OBI"] = np.divide(
        df_bid_ask["bid"] - df_bid_ask["ask"],
        denominator,
        out=np.zeros(len(df_bid_ask)),
        where=denominator != 0
    )

    return df_bid_ask


def process_parquet_files(
    input_dir: Path,
    output_dir: Path,
    list_levels: list[int],
) -> list[Path]:
    """Calculate and cache OBI for all Parquet files in the input directory."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    parquet_files = sorted(input_dir.glob("*.parquet"))

    if not parquet_files:
        raise FileNotFoundError(f"No Parquet files found in {input_dir}")

    if not list_levels or any(level < 0 or level > 9 for level in list_levels):
        raise ValueError("All levels must be between 0 and 9, and the list cannot be empty.")

    if len(list_levels) != len(set(list_levels)):
        raise ValueError("Levels must be unique.")

    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []

    for file in parquet_files:
        levels_label = "-".join(str(level) for level in list_levels)
        output_path = output_dir / f"{file.stem}_OBI_levels_{levels_label}.parquet"

        if output_path.exists():
            print(f"Using cached OBI data: {output_path}", flush=True)
            saved_paths.append(output_path)
            continue

        print(f"Processing: {file.name}", flush=True)

        columns = ["ts_event"] + [
            f"{side}_sz_{level:02d}"
            for level in list_levels
            for side in ("bid", "ask")
        ]

        df = pd.read_parquet(file, columns=columns)

        obi_df = get_OBI(df, list_levels)

        obi_df.to_parquet(output_path, index=False)

        print(f"Saved OBI data: {output_path}", flush=True)

        saved_paths.append(output_path)

        del df, obi_df

    return saved_paths


def main() -> None:
    """Process all raw MBP-10 Parquet files using the default configuration."""
    process_parquet_files(
        input_dir=DEFAULT_INPUT_DIR,
        output_dir=DEFAULT_OUTPUT_DIR,
        list_levels=DEFAULT_LEVELS,
    )


if __name__ == "__main__":
    main()
