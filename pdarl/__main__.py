"""Console entry point: ``python -m pdarl`` or ``pdarl-run``."""

import argparse

from pdarl.trainer import Trainer
from pdarl.utils.args import load_args_from_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PDA training with a YAML config")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help=(
            "Path to a YAML config file (example configs: "
            "https://github.com/JGIoA/pdarl/tree/main/config)"
        ),
    )
    cli_args = parser.parse_args()

    args = load_args_from_yaml(cli_args.config)
    trainer = Trainer(args)
    trainer.setup()
    trainer.run()


if __name__ == "__main__":
    main()
