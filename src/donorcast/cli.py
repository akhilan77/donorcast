import argparse
import sys


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="donorcast",
        description="DonorCast: Forecasting daily blood donations and shortfall alerts.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    from donorcast.clean import clean_data

    def handle_clean(args):
        print("Running data cleaning and verification...")
        _, summary = clean_data()
        print(f"Data cleaning completed successfully. Output rows: {summary['output_rows']:,}")
        print("Saved data/processed/long.parquet and reports/cleaning_log.md")

    subcommands = [
        ("clean", "Clean raw data and produce processed long parquet format.", handle_clean),
        ("features", "Generate time series and calendar features.", None),
        ("baselines", "Run baseline models (M0, M0b) on validation data.", None),
        ("train", "Train models (SARIMA, LightGBM, LSTM).", None),
        ("final", "Run final evaluation on test set.", None),
        ("alerts", "Generate shortfall alerts table.", None),
        ("all", "Run the entire end-to-end pipeline.", None),
    ]

    for cmd, help_text, handler in subcommands:
        subparser = subparsers.add_parser(cmd, help=help_text)
        if handler:
            subparser.set_defaults(func=handler)
        else:
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
