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
        elif args.model in ("lightgbm", "lgbm"):
            print("Training LightGBM model on validation split...")
            from donorcast.models.lgbm import run_lgbm_evaluation

            res = run_lgbm_evaluation(split="val")
            print(f"LightGBM evaluation complete. Version saved: {res['version_dir']}")
        elif args.model == "lstm":
            print("Training LSTM model on validation split...")
            from donorcast.models.lstm import run_lstm_evaluation

            res = run_lstm_evaluation(split="val")
            print(f"LSTM evaluation complete. Version saved: {res['version_dir']}")
        else:
            print(f"Unknown model: {args.model}")

    def handle_final(args):
        print("Starting final model retraining and held-out test evaluation...")
        from donorcast.final import run_final_evaluation

        try:
            run_final_evaluation(force=args.force)
        except RuntimeError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    def handle_explain(args):
        print("Generating global SHAP summary visualizations...")
        from donorcast.explain import generate_global_shap_summary

        res = generate_global_shap_summary(sample_size=args.sample_size)
        print("SHAP summary plots generated successfully:")
        print(f" - Beeswarm: {res['beeswarm']}")
        print(f" - Bar plot: {res['bar']}")

    subcommands = [
        ("clean", "Clean raw data and produce processed long parquet format.", handle_clean),
        ("features", "Generate time series and calendar features.", handle_features),
        ("baselines", "Run baseline models (M0, M0b) on validation data.", handle_baselines),
        ("train", "Train models (SARIMA, LightGBM, LSTM).", handle_train),
        ("final", "Run final evaluation on test set.", handle_final),
        ("explain", "Generate global SHAP summary plots and explainability artifacts.", handle_explain),
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
                choices=["sarima", "lightgbm", "lgbm", "lstm"],
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
        elif cmd == "explain":
            subparser.add_argument(
                "--sample-size",
                type=int,
                default=1000,
                help="Number of test rows to sample for SHAP summary (default: 1000).",
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
