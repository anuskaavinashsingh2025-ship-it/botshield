"""
BotShield – ML Training, Evaluation, and Quality CLI Utility (Phase 8)

Provides a unified command-line interface for:
- Inspecting model artifact status and metadata (--status)
- Displaying the 21-feature contract specification (--contract)
- Training the Isolation Forest anomaly detector on synthetic or DB data (--train)
- Evaluating detection quality metrics against held-out synthetic scenarios (--evaluate)

Usage examples:
    # 1. View 21-feature contract:
    py -m backend.ml.train --contract

    # 2. Inspect active model status:
    py -m backend.ml.train --status

    # 3. Train on synthetic baseline:
    py -m backend.ml.train --train --synthetic --samples 150 --out models/isolation_forest.joblib

    # 4. Evaluate trained model quality metrics:
    py -m backend.ml.train --evaluate --model models/isolation_forest.joblib
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

from backend.database.db import get_connection
from backend.ml.evaluation import evaluate_model_pipeline
from backend.ml.features import (
    FEATURE_CONTRACT,
    FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
    extract_session_features,
)
from backend.ml.model import (
    MLModelStatus,
    get_model_path,
    load_model,
    train_unsupervised_model,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("botshield.ml.train")


def generate_synthetic_baseline(sample_count: int = 100, random_state: int = 42) -> list[list[float]]:
    """Generate representative synthetic human-like session feature vectors for bootstrapping."""
    rng = np.random.RandomState(random_state)
    vectors = []
    for _ in range(sample_count):
        reqs = int(rng.randint(2, 20))
        telems = int(rng.randint(5, 50))
        duration = float(rng.uniform(10.0, 300.0))
        req_rate = reqs / duration
        telem_rate = telems / duration
        telem_ratio = telems / max(reqs, 1)
        req_ratio = reqs / (reqs + telems)
        unique_paths = float(rng.randint(2, min(reqs + 1, 8)))
        path_rep = 1.0 - (unique_paths / reqs)
        pv_count = float(rng.randint(1, min(reqs, 6)))
        clicks = float(rng.randint(2, 15))
        scrolls = float(rng.randint(2, 20))
        mouse = float(rng.randint(5, 40))
        keyboard = float(rng.randint(0, 10))
        page_exits = 1.0
        mean_dt = duration / (reqs + telems)
        std_dt = float(rng.uniform(0.5, 5.0))
        min_dt = float(rng.uniform(0.2, 2.0))
        rapid_count = 0.0
        error_ratio = float(rng.choice([0.0, 0.0, 0.05]))

        vec = [
            float(reqs),
            float(telems),
            float(reqs + telems),
            duration,
            req_rate,
            telem_rate,
            telem_ratio,
            req_ratio,
            unique_paths,
            path_rep,
            pv_count,
            clicks,
            scrolls,
            mouse,
            keyboard,
            page_exits,
            mean_dt,
            std_dt,
            min_dt,
            rapid_count,
            error_ratio,
        ]
        vectors.append(vec)
    return vectors


def extract_vectors_from_db(db_path: str) -> list[list[float]]:
    """Extract 21-dimensional feature vectors for all sessions present in the database."""
    conn = get_connection(db_path)
    try:
        session_rows = conn.execute("SELECT session_id FROM sessions").fetchall()
        vectors = []
        for row in session_rows:
            sess_id = row["session_id"]
            events = conn.execute("SELECT * FROM events WHERE session_id = ?", (sess_id,)).fetchall()
            requests = conn.execute("SELECT * FROM requests WHERE session_id = ?", (sess_id,)).fetchall()
            if events or requests:
                feat = extract_session_features(events=events, requests=requests)
                vectors.append(feat.feature_vector)
        return vectors
    finally:
        conn.close()


def print_feature_contract():
    """Print the 21-feature contract specification."""
    print("=" * 80)
    print(f"BotShield Feature Contract (Schema Version: {FEATURE_SCHEMA_VERSION})")
    print(f"Total Features: {len(FEATURE_NAMES)}")
    print("=" * 80)
    for name in FEATURE_NAMES:
        info = FEATURE_CONTRACT.get(name, {})
        idx = info.get("index", "?")
        bounds = info.get("bounds", ("-", "-"))
        desc = info.get("description", "")
        print(f"[{idx:02d}] {name:<28} Bounds: [{bounds[0]:.1f}, {bounds[1]:.1f}]")
        if desc:
            print(f"     Description: {desc}")
    print("=" * 80)


def print_model_status(model_path: str | Path | None = None):
    """Inspect and print model status and artifact metadata."""
    target_path = get_model_path(model_path)
    print("=" * 80)
    print(f"BotShield Model Artifact Status: {target_path}")
    print("=" * 80)

    bundle, status, warning = load_model(target_path)
    print(f"Status:             {status.value.upper()}")
    print(f"Artifact Exists:    {target_path.exists()}")

    if warning:
        print(f"Notice/Warning:     {warning}")

    if bundle and status == MLModelStatus.AVAILABLE:
        meta = bundle.get("metadata", {})
        print(f"Model Type:         {meta.get('model_type', 'Unknown')}")
        print(f"Schema Version:     {meta.get('schema_version', 'Unknown')}")
        print(f"Trained At:         {meta.get('trained_at', 'Unknown')}")
        print(f"Training Samples:   {meta.get('training_sample_count', 'Unknown')}")
        print(f"Contamination:      {meta.get('contamination', 'Unknown')}")
        print(f"Random State:       {meta.get('random_state', 'Unknown')}")
        print(f"Dataset Type:       {meta.get('dataset_type', 'Unknown')}")
        print(f"Feature Count:      {len(meta.get('feature_names', []))}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="BotShield ML Training & Evaluation CLI (Phase 8)")
    parser.add_argument("--contract", action="store_true", help="Display the 21-feature contract specification.")
    parser.add_argument("--status", action="store_true", help="Inspect and display model artifact status.")
    parser.add_argument("--train", action="store_true", help="Train the Isolation Forest model.")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate model quality on held-out synthetic test data.")

    # Training parameters
    parser.add_argument("--synthetic", action="store_true", help="Generate synthetic human baseline data for training.")
    parser.add_argument(
        "--realistic", action="store_true",
        help="Train/evaluate on pipeline-faithful simulated sessions (backend.ml.realistic); "
             "--samples is then the number of human sessions.",
    )
    parser.add_argument("--samples", type=int, default=100, help="Number of synthetic samples to generate (default 100).")
    parser.add_argument("--db", type=str, default=None, help="Database path to extract training sessions from.")
    parser.add_argument("--out", type=str, default=None, help="Destination model path.")
    parser.add_argument("--model", type=str, default=None, help="Model path for status or evaluation.")
    parser.add_argument("--contamination", type=float, default=0.10, help="Isolation Forest contamination factor.")
    parser.add_argument("--min-samples", type=int, default=10, help="Minimum sample count requirement.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--threshold", type=float, default=None, help="Operating threshold for evaluation.")

    args = parser.parse_args()

    # If no flags passed, print help and exit
    if not (args.contract or args.status or args.train or args.evaluate or args.synthetic or args.db or args.realistic):
        parser.print_help()
        sys.exit(0)

    if args.contract:
        print_feature_contract()
        sys.exit(0)

    if args.status:
        print_model_status(args.model or args.out)
        sys.exit(0)

    if args.evaluate and args.realistic:
        from backend.ml.realistic import evaluate_model_realistic

        target_path = get_model_path(args.model or args.out)
        report = evaluate_model_realistic(str(target_path))
        res = report["results"]
        cm, met = res["confusion_matrix"], res["metrics"]
        print("\n" + "=" * 80)
        print(report["evaluation_name"])
        print("=" * 80)
        print(f"Model Artifact:      {target_path}")
        print(f"Operating Threshold: {report['operating_threshold']} (model's own decision boundary)")
        print(f"TP {cm['tp']}  FP {cm['fp']}  TN {cm['tn']}  FN {cm['fn']}")
        print(f"Precision {met['precision']:.4f}  Recall {met['recall']:.4f}  "
              f"F1 {met['f1_score']:.4f}  FPR {met['false_positive_rate']:.4f}")
        for kind, b in res["per_scenario"].items():
            print(f"  {kind:<11} flagged {b['flagged']:>3} / {b['sessions']}")
        print(f"\n{report['disclaimer']}\n")
        sys.exit(0)

    if args.evaluate:
        target_path = get_model_path(args.model or args.out)
        logger.info("Evaluating model artifact: %s", target_path)
        try:
            report = evaluate_model_pipeline(
                model_path=target_path,
                random_state=args.seed,
                threshold=args.threshold,
            )
            print("\n" + "=" * 80)
            print(f"DETECTION QUALITY EVALUATION REPORT")
            print("=" * 80)
            print(f"Model Artifact:      {report['model_artifact']}")
            meta = report.get("model_metadata", {})
            if meta:
                print(f"Model Version:       {meta.get('schema_version', 'Unknown')}")
                print(f"Dataset Type:        {meta.get('dataset_type', 'Unknown')}")
                print(f"Training Samples:    {meta.get('training_sample_count', 'Unknown')}")
                print(f"Contamination:       {meta.get('contamination', 'Unknown')}")
                print(f"Random State:        {meta.get('random_state', 'Unknown')}")
                print(f"Trained At:          {meta.get('trained_at', 'Unknown')}")
            print(f"Evaluation Name:     {report['evaluation_name']}")
            print(f"Real-World Claim:    {report['real_world_accuracy_claim']} (Controlled Synthetic Baseline)")
            print(f"Operating Threshold: {report['operating_threshold']:.4f}")
            print("-" * 80)
            res = report["results"]
            cm = res["confusion_matrix"]
            print(f"Confusion Matrix:")
            print(f"  True Positives (TP):  {cm['tp']:>5}  |  False Positives (FP): {cm['fp']:>5}")
            print(f"  False Negatives (FN): {cm['fn']:>5}  |  True Negatives (TN):  {cm['tn']:>5}")
            print("-" * 80)
            met = res["metrics"]
            print(f"Precision:           {met['precision']:.4f}")
            print(f"Recall:              {met['recall']:.4f}")
            print(f"F1 Score:            {met['f1_score']:.4f}")
            print(f"False Positive Rate: {met['false_positive_rate']:.4f}")
            print(f"False Negative Rate: {met['false_negative_rate']:.4f}")
            print(f"Accuracy:            {met['accuracy']:.4f}")
            print("=" * 80)
            print(f"\n{report['disclaimer']}\n")
            sys.exit(0)
        except Exception as exc:
            logger.error("Evaluation failed: %s", exc)
            sys.exit(1)

    if args.train or args.synthetic or args.db or args.realistic:
        training_vectors = []
        dataset_type = "synthetic_baseline"
        if args.realistic:
            from backend.ml.realistic import build_training_vectors

            logger.info("Generating pipeline-faithful human sessions (%d sessions, seed=%d)...", args.samples, args.seed)
            training_vectors = build_training_vectors(n_sessions=args.samples, seed=args.seed)
            dataset_type = f"realistic_pipeline_{args.samples}"
        elif args.synthetic or (not args.db and args.train):
            logger.info("Generating synthetic training baseline (%d samples, seed=%d)...", args.samples, args.seed)
            training_vectors = generate_synthetic_baseline(sample_count=args.samples, random_state=args.seed)
            dataset_type = f"synthetic_baseline_{args.samples}"
        elif args.db:
            logger.info("Extracting session features from database '%s'...", args.db)
            if not Path(args.db).exists():
                logger.error("Database file '%s' does not exist.", args.db)
                sys.exit(1)
            training_vectors = extract_vectors_from_db(args.db)
            dataset_type = f"sqlite_db_{Path(args.db).name}"
        else:
            logger.error("Must specify training data source via --synthetic or --db <path>.")
            sys.exit(1)

        if len(training_vectors) < args.min_samples:
            logger.error(
                "Insufficient samples for training: found %d, minimum required is %d.",
                len(training_vectors),
                args.min_samples,
            )
            sys.exit(1)

        target_path = get_model_path(args.out)
        try:
            metadata = train_unsupervised_model(
                training_data=training_vectors,
                model_path=target_path,
                contamination=args.contamination,
                random_state=args.seed,
                min_samples=args.min_samples,
                dataset_type=dataset_type,
                # Pipeline sessions have heavy-tailed counts/durations
                log_transform=bool(args.realistic),
                n_estimators=200 if args.realistic else 100,
            )
            logger.info("Training complete. Artifact saved atomically to: %s", target_path)
            logger.info("Metadata: %s", metadata)
            sys.exit(0)
        except Exception as exc:
            logger.error("Training failed: %s", exc)
            sys.exit(1)


if __name__ == "__main__":
    main()
