#!/usr/bin/env python
"""
Main training script for Bangla Embedding Model.

Usage:
    python scripts/train.py --config configs/training_config.yaml --mode multi_stage
"""

import argparse
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_preparation import prepare_training_data
from src.trainer import train_bangla_embedding_model, BanglaEmbeddingTrainer, TrainingConfig
from src.evaluate import run_full_evaluation

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train Bangla Embedding Model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training_config.yaml",
        help="Path to training configuration file"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="multi_stage",
        choices=["standard", "distillation", "multi_stage"],
        help="Training mode"
    )
    parser.add_argument(
        "--prepare_data",
        action="store_true",
        help="Prepare training data before training"
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run MTEB evaluation after training"
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Push model to Hugging Face Hub after training"
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="Hugging Face Hub model ID (e.g., 'username/model-name')"
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Fast mode for testing: limit to 10k samples"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to use (overrides --fast)"
    )
    
    args = parser.parse_args()
    
    # Determine max samples
    max_samples = args.max_samples
    if args.fast and max_samples is None:
        max_samples = 10000  # Default fast mode limit
    
    # Step 1: Prepare data if requested
    if args.prepare_data:
        logger.info("Preparing training data...")
        prepare_training_data(
            output_dir="./data/processed",
            include_parallel=False,  # Skip parallel (not available)
            include_paraphrase=True,
            include_nli=False,  # Skip NLI (not available)
            max_samples=max_samples
        )
    
    # Step 2: Train model
    logger.info(f"Starting training with mode: {args.mode}")
    model = train_bangla_embedding_model(args.config, args.mode)
    
    # Get output directory from config
    trainer = BanglaEmbeddingTrainer(args.config)
    output_dir = trainer.config.output_dir
    model_path = str(Path(output_dir) / "final")
    
    # Step 3: Evaluate if requested
    if args.evaluate:
        logger.info("Running MTEB evaluation...")
        results = run_full_evaluation(
            model_path=model_path,
            output_dir=str(Path(output_dir) / "evaluation")
        )
        logger.info(f"Evaluation complete. Results: {results}")
    
    # Step 4: Push to Hub if requested
    if args.push_to_hub:
        if args.hub_model_id is None:
            logger.error("--hub_model_id is required when using --push_to_hub")
            return
        
        # Get HF token from environment
        hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
        if not hf_token:
            logger.warning("No HF_TOKEN found in environment. Make sure you're logged in via `huggingface-cli login` or set HF_TOKEN in .env file")
        
        logger.info(f"Pushing model to Hugging Face Hub: {args.hub_model_id}")
        model.push_to_hub(args.hub_model_id, token=hf_token)
        logger.info("Model pushed successfully!")
    
    logger.info("Training pipeline complete!")


if __name__ == "__main__":
    main()