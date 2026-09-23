import argparse
import sys


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="donorcast",
        description="DonorCast: Forecasting daily blood donations and shortfall alerts.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    subcommands = [
        ("clean", "Clean raw data and produce processed long parquet format."),
        ("features", "Generate time series and calendar features."),
        ("baselines", "Run baseline models (M0, M0b) on validation data."),
        ("train", "Train models (SARIMA, LightGBM, LSTM)."),
        ("final", "Run final evaluation on test set."),
        ("alerts", "Generate shortfall alerts table."),
        ("all", "Run the entire end-to-end pipeline."),
    ]

    for cmd, help_text in subcommands:
        subparser = subparsers.add_parser(cmd, help=help_text)
        subparser.set_defaults(func=lambda args, c=cmd: print(f"Placeholder: Subcommand '{c}' called."))

    return parser


def main():
    parser = create_parser()
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
