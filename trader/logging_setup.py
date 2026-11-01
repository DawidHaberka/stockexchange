"""Logi: konsola + plik rotowany runtime/logs/trader.log."""

from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path


def setup_logging(log_dir: str | Path, verbose: bool = False) -> None:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    for h in list(root.handlers):
        root.removeHandler(h)
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    root.addHandler(console)
    fh = logging.handlers.RotatingFileHandler(Path(log_dir) / "trader.log", maxBytes=5_000_000, backupCount=10,
                                              encoding="utf-8")
    fh.setFormatter(fmt)
    root.addHandler(fh)
    # biblioteki HTTP są gadatliwe
    for noisy in ("urllib3", "matplotlib", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
