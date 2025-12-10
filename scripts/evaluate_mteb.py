#!/usr/bin/env python
"""
MTEB Evaluation Script for Bangla Embedding Models.

Usage:
    python scripts/evaluate_mteb.py --model_path ./outputs/bangla-embedding/final
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluate import BanglaEmbeddingEvaluator, run_full_evaluation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Bangla model on MTEB")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained model or Hugging Face model ID"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./evaluation_results",
        help="Directory to save evaluation results"
    )
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=None,
        help="Specific MTEB tasks to evaluate (default: all Bengali tasks)"
    )
    parser.add_argument(
        "--generate_submission",
        action="store_true",
        help="Generate MTEB leaderboard submission files"
    )
    
    args = parser.parse_args()
    
    evaluator = BanglaEmbeddingEvaluator(
        model_path=args.model_path,
        output_dir=args.output_dir
    )
    
    # Run evaluation
    if args.tasks:
        import mteb
        tasks = mteb.get_tasks(tasks=args.tasks)
        evaluation = mteb.MTEB(tasks=tasks)
        results = evaluation.run(
            evaluator.model,
            output_folder=args.output_dir
        )
    else:
        results = evaluator.evaluate_on_bengali()
    
    # Print summary
    print("\n" + "="*50)
    print("EVALUATION RESULTS SUMMARY")
    print("="*50)
    
    if 'summary' in results:
        for task_type, scores in results['summary'].items():
            print(f"\n{task_type}:")
            print(f"  Average Score: {scores.get('average', 'N/A'):.4f}")
            print(f"  Number of Tasks: {scores.get('num_tasks', 'N/A')}")
    
    # Generate submission files
    if args.generate_submission:
        submission = evaluator.generate_leaderboard_submission()
        print(f"\nSubmission files saved to: {submission['submission_dir']}")
    
    # Save full results
    results_file = Path(args.output_dir) / "full_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nFull results saved to: {results_file}")


if __name__ == "__main__":
    main()