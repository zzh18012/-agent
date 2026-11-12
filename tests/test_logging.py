import json
import logging
from pathlib import Path

import structlog

from mmos.utils.logging import configure_logging


def test_json_log_file_receives_structured_events(tmp_path: Path) -> None:
    log_file = tmp_path / "logs" / "mmos.jsonl"
    configure_logging("INFO", json_file=log_file, console=False)

    structlog.get_logger("test").info("hello", value=1)
    logging.shutdown()

    lines = log_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    event = json.loads(lines[0])
    assert event["event"] == "hello"
    assert event["value"] == 1
    assert event["level"] == "info"
    assert "timestamp" in event
