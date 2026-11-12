import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from mmos.core.project import ProjectPaths, atomic_write_text, load_experiment_spec
from mmos.data import DataAgent


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the data-quality gate for an experiment.")
    parser.add_argument("--project", required=True, help="Project directory.")
    parser.add_argument("--experiment-id", required=True)
    args = parser.parse_args(argv)

    paths = ProjectPaths(Path(args.project).resolve())
    spec = load_experiment_spec(paths, args.experiment_id)
    result = DataAgent(paths).analyze(target=spec.target_column)

    artifacts = paths.experiments_dir / args.experiment_id / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment_id": args.experiment_id,
        "target_column": spec.target_column,
        "quality": result.quality.model_dump(mode="json"),
        "leakage": result.leakage.model_dump(mode="json"),
    }
    atomic_write_text(artifacts / "data_gate.json", json.dumps(payload, indent=2) + "\n")

    if result.quality.blocking or result.leakage.blocking:
        print("data gate failed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
