"""Command-line entry point for Colosseum inference and custom training."""

from __future__ import annotations

import sys

from . import prediction, training


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help"}:
        print("usage: colosseum {predict,train} [options]")
        print("Use 'colosseum predict --help' or 'colosseum train --help'.")
        return
    command, *rest = args
    if command == "predict":
        prediction.main(rest)
    elif command == "train":
        training.main(rest)
    else:
        raise SystemExit(f"unknown command: {command}. Use 'colosseum --help'.")
