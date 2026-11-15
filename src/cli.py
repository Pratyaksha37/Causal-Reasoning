"""Probabilistic modelling command-line interface; run `python -m src --help`."""

import argparse
import json
import sys
from pathlib import Path

import requests


def assignments(items: list[str]) -> dict[str, str]:
    result = {}
    for item in items:
        if "=" not in item:
            raise ValueError("Assignments must use Variable=state")
        variable, state = item.split("=", 1)
        if not variable or not state or variable in result:
            raise ValueError("Assignments must be nonempty and unique")
        result[variable] = state
    return result


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "extract":
        from run_pipeline import main as extraction_main

        sys.argv = [sys.argv[0], *sys.argv[2:]]
        extraction_main()
        return 0
    parser = argparse.ArgumentParser(
        description="Diabetes evidence: NLP → DAG → Bayesian estimation → reasoning"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser(
        "run", help="Run the complete integrated evidence and reasoning project"
    )
    run_parser.add_argument("--config", default="configs/bayesian.yaml")
    run_parser.add_argument("--data")
    run_parser.add_argument("--output")
    run_parser.add_argument("--demo", action="store_true")
    run_parser.add_argument("--reviews", help="Explicit human submissions (CSV or JSONL)")
    run_parser.add_argument("--adjudications", help="Senior human adjudications (JSONL)")
    run_parser.add_argument(
        "--skip-transformer",
        action="store_true",
        help="Explicitly omit transformer evaluation if weights are unavailable",
    )
    run_parser.add_argument("--benchmark-config", default="configs/benchmark.yaml")
    sub.add_parser(
        "extract", help="Legacy corpus extraction options, including --from-saved-predictions"
    )
    bench = sub.add_parser(
        "benchmark", help="Controlled document-holdout rule/ML/transformer comparison"
    )
    bench.add_argument("--config", default="configs/benchmark.yaml")
    bench.add_argument("--skip-transformer", action="store_true")
    prepare = sub.add_parser("prepare-encoder", help="Prepare the pinned real pretrained encoder")
    prepare.add_argument("--allow-download", action="store_true")
    predict = sub.add_parser("predict-text", help="Extract relations from a supplied sentence")
    predict.add_argument("--text", required=True)
    predict.add_argument("--rules-only", action="store_true")
    for name in ["query", "intervene"]:
        item = sub.add_parser(name)
        item.add_argument("--model", required=True)
        item.add_argument("--target", default="Outcome")
        item.add_argument("--evidence", nargs="*", default=[])
        if name == "intervene":
            item.add_argument("--do", dest="interventions", nargs="+", required=True)
            item.add_argument("--assume-causal-sufficiency", action="store_true")
            item.add_argument("--assume-consistency", action="store_true")
            item.add_argument("--acknowledge-structural-prior", action="store_true")
    review = sub.add_parser("review-report")
    review.add_argument("--queue", default="data/annotations/human_annotation_queue.jsonl")
    review.add_argument("--reviews")
    review.add_argument("--adjudications")
    review.add_argument("--export-form")
    download = sub.add_parser("download", help="Explicitly download official public BRFSS 2015")
    download.add_argument("--output", default="data/brfss/raw/LLCP2015XPT.zip")
    args = parser.parse_args()
    result: dict
    try:
        if args.command == "download":
            from src.data.brfss import DOWNLOAD_URL, file_sha256

            target = Path(args.output)
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_suffix(target.suffix + ".partial")
            try:
                with requests.get(DOWNLOAD_URL, timeout=(20, 60), stream=True) as response:
                    response.raise_for_status()
                    with partial.open("wb") as stream:
                        for chunk in response.iter_content(1024 * 1024):
                            stream.write(chunk)
                import zipfile

                with zipfile.ZipFile(partial) as archive:
                    if archive.testzip() is not None:
                        raise ValueError("Corrupt BRFSS archive")
                partial.replace(target)
            finally:
                partial.unlink(missing_ok=True)
            result = {"path": str(target), "source": DOWNLOAD_URL, "sha256": file_sha256(target)}
        elif args.command == "run":
            from src.pipeline import run_project

            project = run_project(
                args.config,
                args.data,
                args.output,
                args.demo,
                args.reviews,
                args.adjudications,
                not args.skip_transformer,
                args.benchmark_config,
            )
            summary = project["bayesian"]
            result = {
                "model_fitted": summary["model_fitted"],
                "source": summary["source"],
                "rows": summary["preprocessing"]["rows_retained"],
                "evaluation": summary["evaluation"],
                "human_validation": summary["human_validation"],
                "causal_validity_established": False,
            }
        elif args.command == "benchmark":
            from src.evaluation.benchmark import benchmark

            result = benchmark(args.config, include_transformer=not args.skip_transformer)
        elif args.command == "prepare-encoder":
            from src.extractors.transformer_encoder import prepare_encoder
            from src.utils.config import load_yaml_config

            result = prepare_encoder(
                load_yaml_config("configs/benchmark.yaml")["encoder"], args.allow_download
            )
        elif args.command == "predict-text":
            from src.extractors.learned_classifier import extract_text

            result = extract_text(args.text, not args.rules_only)
        elif args.command == "review-report":
            from src.annotation.human_review import (
                Adjudication,
                adjudicate,
                export_review_form,
                read_reviews,
            )

            decisions = (
                [
                    Adjudication.model_validate_json(line)
                    for line in Path(args.adjudications).read_text().splitlines()
                    if line.strip()
                ]
                if args.adjudications
                else []
            )
            result = adjudicate(args.queue, read_reviews(args.reviews), decisions)
            if args.export_form:
                export_review_form(args.queue, args.export_form)
        else:
            from src.bayesian.inference import query
            from src.bayesian.network import load_model
            from src.causal.intervention import intervene

            bundle = load_model(args.model)
            if args.command == "query":
                result = query(bundle, args.target, assignments(args.evidence))
            else:
                assumptions = {
                    key: getattr(args, key)
                    for key in [
                        "assume_causal_sufficiency",
                        "assume_consistency",
                        "acknowledge_structural_prior",
                    ]
                }
                result = intervene(
                    bundle,
                    args.target,
                    assignments(args.interventions),
                    assignments(args.evidence),
                    assumptions,
                )
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    except (ValueError, OSError, KeyError, requests.RequestException) as error:
        print(f"Probabilistic modelling error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
