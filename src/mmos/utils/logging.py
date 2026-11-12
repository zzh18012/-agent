import logging
from pathlib import Path

import structlog

_SHARED_PROCESSORS: list[structlog.typing.Processor] = [
    structlog.contextvars.merge_contextvars,
    structlog.processors.add_log_level,
    structlog.processors.TimeStamper(fmt="iso", utc=True),
    structlog.processors.StackInfoRenderer(),
    structlog.processors.format_exc_info,
]


def configure_logging(
    level: str = "INFO",
    *,
    json_file: Path | None = None,
    console: bool = True,
) -> None:
    """Route structlog to a human-readable console and an optional JSONL file.

    Human-readable output is for the developer watching the terminal; the JSONL
    file is the machine-readable record that survives the session.
    """
    handlers: list[logging.Handler] = []

    if console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(
            structlog.stdlib.ProcessorFormatter(
                processor=structlog.dev.ConsoleRenderer(),
                foreign_pre_chain=_SHARED_PROCESSORS,
            )
        )
        handlers.append(console_handler)

    if json_file is not None:
        json_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(json_file, encoding="utf-8")
        file_handler.setFormatter(
            structlog.stdlib.ProcessorFormatter(
                processor=structlog.processors.JSONRenderer(),
                foreign_pre_chain=_SHARED_PROCESSORS,
            )
        )
        handlers.append(file_handler)

    logging.basicConfig(level=level.upper(), handlers=handlers, format="%(message)s", force=True)

    structlog.configure(
        processors=[
            *_SHARED_PROCESSORS,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
