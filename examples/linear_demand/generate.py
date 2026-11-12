"""Generate the synthetic demand demonstration dataset and its metadata.

Run from the repository root:
    uv run python examples/linear_demand/generate.py --out scratch/linear_demand
"""

import argparse
import json
import math
from pathlib import Path

import polars as pl

SEED = 20261112
PERIODS = 60


def build_frame(periods: int = PERIODS) -> pl.DataFrame:
    period = list(range(1, periods + 1))
    x1 = [float((value * 3) % 10) for value in period]
    x2 = [float((value * 5) % 7) for value in period]
    y = [
        10.0 + 3.0 * a + 2.0 * b + 0.4 * math.sin(value)
        for value, a, b in zip(period, x1, x2, strict=True)
    ]
    return pl.DataFrame({"period": period, "x1": x1, "x2": x2, "y": y})


def metadata() -> dict:
    return {
        "schema_version": 1,
        "target_name": "y",
        "target_description": "product demand per period (synthetic)",
        "target_unit": None,
        "unit_status": "synthetic_units",
        "temporal": {
            "is_temporal": True,
            "time_column": "period",
            "frequency": "1 period",
            "forecast_horizon": 1,
            "seasonal_period": None,
            "ordering_required": True,
            "future_feature_policy": "known_future_only",
        },
        "features": [
            {"name": "x1", "availability": "known_future"},
            {"name": "x2", "availability": "known_future"},
        ],
        "provenance": {
            "is_synthetic": True,
            "generator": "linear_demand_generator_v1",
            "generation_parameters": {
                "periods": PERIODS,
                "equation": "y = 10 + 3*x1 + 2*x2 + 0.4*sin(period)",
                "x1": "(3*period) mod 10",
                "x2": "(5*period) mod 7",
            },
            "seed": SEED,
            "code_path": "examples/linear_demand/generate.py",
            "description": "Synthetic demonstration data; not real-world demand.",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    build_frame().write_csv(args.out / "linear_demand.csv")
    (args.out / "dataset_metadata.json").write_text(
        json.dumps(metadata(), indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {args.out / 'linear_demand.csv'}")
    print(f"wrote {args.out / 'dataset_metadata.json'}")


if __name__ == "__main__":
    main()
