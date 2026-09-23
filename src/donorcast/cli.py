import argparse
import sys


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="donorcast",
        description="DonorCast: Forecasting daily blood donations and shortfall alerts.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    from donorcast.clean import clean_data
    from donorcast.features import generate_all_feature_datasets

    def handle_clean(args):
        print("Running data cleaning and verification...")
        _, summary = clean_data()
        print(f"Data cleaning completed successfully. Output rows: {summary['output_rows']:,}")
        print("Saved data/processed/long.parquet and reports/cleaning_log.md")

    def handle_features(args):
        print("Running multi-horizon feature generation...")
        saved_files = generate_all_feature_datasets()
        print(f"Feature generation completed. {len(saved_files)} files saved.")

    def handle_baselines(args):
        print("Running baseline models (M0, M0b) on validation split...")
        from donorcast.models.baselines import run_baselines_evaluation

        res = run_baselines_evaluation(split="val")
        print(f"Baselines evaluation complete. Report written to: {res['summary_file']}")

    def handle_train(args):
        if args.model == "sarima":
            print("Training SARIMAX model on validation split...")
            from donorcast.models.sarima import run_sarima_evaluation

            res = run_sarima_evaluation(split="val", n_jobs=args.n_jobs)
            print(f"SARIMAX evaluation complete. Report written to: {res['report_file']}")
        elif args.model == "lightgbm":
            print("LightGBM model training will be implemented in Task 3.2.")
        elif args.model == "lstm":
            print("LSTM model training will be implemented in Task 3.3.")
        else:
            print(f"Unknown model: {args.model}")

    def handle_final(args):
        print("Running final evaluation on test set...")
        from donorcast.evaluate import FINAL_RUN_FILE

        if FINAL_RUN_FILE.exists() and not args.force:
            print(
                f"Error: Final test set run already completed (found {FINAL_RUN_FILE}). "
                "Use --force to run again.",
                file=sys.stderr,
            )
            sys.exit(1)
        print("Final evaluation mode enabled. (Use evaluate with allow_test=True)")

    subcommands = [
        ("clean", "Clean raw data and produce processed long parquet format.", handle_clean),
        ("features", "Generate time series and calendar features.", handle_features),
        ("baselines", "Run baseline models (M0, M0b) on validation data.", handle_baselines),
        ("train", "Train models (SARIMA, LightGBM, LSTM).", handle_train),
        ("final", "Run final evaluation on test set.", handle_final),
        ("alerts", "Generate shortfall alerts table.", None),
        ("all", "Run the entire end-to-end pipeline.", None),
    ]

    for cmd, help_text, handler in subcommands:
        subparser = subparsers.add_parser(cmd, help=help_text)
        if cmd == "train":
            subparser.add_argument(
                "--model",
                type=str,
                default="sarima",
                choices=["sarima", "lightgbm", "lstm"],
                help="Model to train and evaluate on validation split.",
            )
            subparser.add_argument(
                "--n-jobs",
                type=int,
                default=-1,
                help="Number of CPU cores for parallelization (-1 = all cores).",
            )
        elif cmd == "final":
            subparser.add_argument(
                "--force",
                action="store_true",
                help="Force re-running the final test evaluation even if final_run.json exists.",
            )
        if handler:
            subparser.set_defaults(func=handler)
        else:
            subparser.set_defaults(
                func=lambda args, c=cmd: print(f"Placeholder: Subcommand '{c}' called.")
            )

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
