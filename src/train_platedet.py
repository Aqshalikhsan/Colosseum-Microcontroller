"""Compatibility entry point for the Colosseum training loop."""

from colosseum._core.train_platedet import *  # noqa: F401,F403
from colosseum._core.train_platedet import main


if __name__ == "__main__":
    main()
