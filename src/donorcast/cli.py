import argparse
import sys


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="donorcast",
        description="DonorCast: Forecasting daily blood donations and shortfall alerts.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    def handle_clean(args):
        print("Running data cleaning and verification...")
        from donorcast.clean import clean_data

        _, summary = clean_data()
        print(f"Data cleaning completed successfully. Output rows: {summary['output_rows']:,}")
        print("Saved data/processed/long.parquet and reports/cleaning_log.md")

    def handle_features(args):
        print("Running multi-horizon feature generation...")
        from donorcast.features import generate_all_feature_datasets

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

    def handle_alerts(args):
        from donorcast.shortfall import generate_and_save_alerts, precompute_replay_origins

        if getattr(args, "precompute_replay", False):
            precompute_replay_origins()
            print("\nGenerating live alerts for origin:", args.origin)
            generate_and_save_alerts(origin_date=args.origin)
        else:
            generate_and_save_alerts(origin_date=args.origin)

    def handle_all(args):
        import time

        from donorcast.clean import clean_data
        from donorcast.features import generate_all_feature_datasets
        from donorcast.models.baselines import run_baselines_evaluation
        from donorcast.models.lgbm import run_lgbm_evaluation
        from donorcast.shortfall import generate_and_save_alerts, precompute_replay_origins

        origin = getattr(args, "origin", DATA_CUTOFF)
        precompute_replay = getattr(args, "precompute_replay", False)

        pipeline_start = time.perf_counter()
        timings: list[tuple[str, str, float]] = []

        print("=" * 80)
        print("          DONORCAST END-TO-END PIPELINE ORCHESTRATOR")
        print("=" * 80)
        print(f"Origin Date:            {origin}")
        print(f"Precompute Replay:      {precompute_replay}")
        print(
            "Pipeline Sequence:      1. Clean -> 2. Features -> 3. Baselines -> 4. Train -> 5. Alerts"
        )
        print("Test Set Policy:        Skips 'final' (manual held-out test evaluation only)")
        print("=" * 80 + "\n")

        # Stage 1: Data Cleaning
        print(">>> [1/5] RUNNING DATA CLEANING & RECONCILIATION CHECKS...")
        t0 = time.perf_counter()
        _, summary = clean_data()
        d1 = time.perf_counter() - t0
        timings.append(("1. Data Cleaning & Integrity", "COMPLETED", d1))
        print(
            f"[OK] [1/5] Clean finished in {d1:.2f}s "
            f"({summary['output_rows']:,} rows saved to data/processed/long.parquet)\n"
        )

        # Stage 2: Multi-Horizon Features
        print(">>> [2/5] RUNNING MULTI-HORIZON FEATURE GENERATION...")
        t0 = time.perf_counter()
        saved_features = generate_all_feature_datasets()
        d2 = time.perf_counter() - t0
        timings.append(("2. Multi-Horizon Features", "COMPLETED", d2))
        print(
            f"[OK] [2/5] Features finished in {d2:.2f}s "
            f"({len(saved_features)} yearly partitions generated)\n"
        )

        # Stage 3: Baseline Models (M0, M0b)
        print(">>> [3/5] RUNNING BASELINE MODELS EVALUATION (M0 & M0b)...")
        t0 = time.perf_counter()
        res_baselines = run_baselines_evaluation(split="val")
        d3 = time.perf_counter() - t0
        timings.append(("3. Baseline Models (M0, M0b)", "COMPLETED", d3))
        print(
            f"[OK] [3/5] Baselines finished in {d3:.2f}s "
            f"(Report: {res_baselines['summary_file']})\n"
        )

        # Stage 4: Train Selected Model (LightGBM + Quantiles)
        print(">>> [4/5] TRAINING SELECTED MODEL (LightGBM + Quantiles p10/p90)...")
        t0 = time.perf_counter()
        res_lgbm = run_lgbm_evaluation(split="val")
        d4 = time.perf_counter() - t0
        timings.append(("4. Train LightGBM + Quantiles", "COMPLETED", d4))
        print(
            f"[OK] [4/5] Training finished in {d4:.2f}s "
            f"(Artifacts saved: {res_lgbm['version_dir']})\n"
        )

        # Stage 5: Alerts & Forecasts
        print(">>> [5/5] GENERATING 14-DAY FORECASTS & SHORTFALL ALERTS...")
        t0 = time.perf_counter()
        if precompute_replay:
            print("Precomputing historical replay origins...")
            precompute_replay_origins()
            print(f"Generating live alerts for origin: {origin}...")
        res_alerts = generate_and_save_alerts(origin_date=origin)
        d5 = time.perf_counter() - t0
        timings.append(("5. Shortfall Alerts & Forecasts", "COMPLETED", d5))
        print(
            f"[OK] [5/5] Alerts finished in {d5:.2f}s "
            f"(Saved {len(res_alerts)} alerts for {origin})\n"
        )

        total_time = time.perf_counter() - pipeline_start
        mins, secs = divmod(total_time, 60)

        # Final Summary Banner
        print("=" * 80)
        print("                    DONORCAST PIPELINE EXECUTION SUMMARY")
        print("=" * 80)
        print(f"{'Pipeline Stage':<38} | {'Status':<10} | {'Duration (s)':>12}")
        print("-" * 80)
        for stage, status, dur in timings:
            print(f"{stage:<38} | {status:<10} | {dur:>10.2f}s")
        print("-" * 80)
        print(
            f"{'TOTAL RUNTIME':<38} | {'SUCCESS':<10} | "
            f"{total_time:>10.2f}s ({int(mins)}m {secs:04.1f}s)"
        )
        print("=" * 80)

    from donorcast.config import DATA_CUTOFF

    subcommands = [
        ("clean", "Clean raw data and produce processed long parquet format.", handle_clean),
        ("features", "Generate time series and calendar features.", handle_features),
        ("baselines", "Run baseline models (M0, M0b) on validation data.", handle_baselines),
        ("train", "Train models (SARIMA, LightGBM, LSTM).", handle_train),
        ("final", "Run final evaluation on test set.", handle_final),
        (
            "explain",
            "Generate global SHAP summary plots and explainability artifacts.",
            handle_explain,
        ),
        ("alerts", "Generate shortfall alerts table and 14-day forecasts.", handle_alerts),
        ("all", "Run the entire end-to-end pipeline.", handle_all),
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
        elif cmd in ("alerts", "all"):
            subparser.add_argument(
                "--origin",
                type=str,
                default=DATA_CUTOFF,
                help=f"Forecast origin date in YYYY-MM-DD format (default: {DATA_CUTOFF}).",
            )
            subparser.add_argument(
                "--precompute-replay",
                action="store_true",
                help="Precompute forecasts and alerts for the 8 historical replay origins.",
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
