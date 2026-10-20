import math
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from src.threshold_analysis import select_files, summarize_results  # noqa: E402


def test_select_files_filters_by_start_date(tmp_path):
    filenames = [
        "CL_c_0_mbp-10_2026-10-04_2026-10-05.parquet",
        "CL_c_0_mbp-10_2026-10-05_2026-10-06.parquet",
        "CL_c_0_mbp-10_2026-10-06_2026-10-07.parquet",
    ]
    for filename in filenames:
        (tmp_path / filename).touch()

    files = select_files(tmp_path, "2026-10-05", "2026-10-07")

    assert [file.name for file in files] == filenames[1:]


def test_summarize_results_sums_daily_runs_without_compounding():
    daily = pd.DataFrame(
        [
            {
                "data_file": "day-1.parquet",
                "threshold": 0.75,
                "book_events": 100,
                "tradable_events": 90,
                "round_trips": 2,
                "wins": 1,
                "gross_pnl": 20.0,
                "fees": 4.0,
                "net_pnl": 16.0,
                "max_drawdown_mtm": 10.0,
                "entries_blocked_no_equity": 0,
            },
            {
                "data_file": "day-2.parquet",
                "threshold": 0.75,
                "book_events": 200,
                "tradable_events": 180,
                "round_trips": 3,
                "wins": 1,
                "gross_pnl": -10.0,
                "fees": 6.0,
                "net_pnl": -16.0,
                "max_drawdown_mtm": 25.0,
                "entries_blocked_no_equity": 2,
            },
        ]
    )

    summary = summarize_results(daily).iloc[0]

    assert summary["daily_files"] == 2
    assert summary["book_events"] == 300
    assert summary["round_trips"] == 5
    assert summary["wins"] == 2
    assert math.isclose(summary["win_rate"], 0.4)
    assert math.isclose(summary["gross_pnl"], 10.0)
    assert math.isclose(summary["fees"], 10.0)
    assert math.isclose(summary["net_pnl"], 0.0)
    assert math.isclose(summary["max_daily_drawdown_mtm"], 25.0)
