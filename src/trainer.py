"""
Training module for Bangla embedding model. 
Supports multiple training strategies including knowledge distillation and Matryoshka learning.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import yaml

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from sentence_transformers import (
    SentenceTransformer,
    InputExample,
    losses,
    evaluation,
    models
)
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments

from .dataset import (
    BanglaEmbeddingDataset,
    HardNegativeMiningDataset,
    DistillationDataset,
    create_dataloader
)

# Import the HF_TOKEN from the .env file
HF_TOKEN = os.getenv("HF_TOKEN")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



@dataclass
class TrainingConfig:
    """Training configuration dataclass."""
    # Model
    base_model:  str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    teacher_model: str = "sentence-transformers/all-mpnet-base-v2"
    max_seq_length: int = 256
    
    # Training
    output_dir: str = "./outputs/bangla-embedding"
    num_epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    fp16: bool = True
    gradient_accumulation_steps: int = 1
    
    # Loss
    loss_type: str = "mnrl"  # mnrl, cosine, mse, contrastive
    use_matryoshka: bool = True
    matryoshka_dims: List[int] = None
    
    # Data
    train_data_path: str = "./data/processed/train.json"
    eval_data_path:  str = "./data/processed/eval.json"
    parallel_data_path: str = "./data/parallel/en_bn_parallel.json"
    
    # Evaluation
    eval_steps: int = 500
    save_steps: int = 1000
    
    # Logging
    use_wandb: bool = True
    project_name: str = "bangla-embedding"
    run_name: str = "bangla-mpnet-v1"
    
    def __post_init__(self):
        if self.matryoshka_dims is None:
            self.matryoshka_dims = [768, 512, 256, 128, 64]


class BanglaEmbeddingTrainer: 
    """
    Main trainer class for Bangla embedding model.
    """
    
    def __init__(self, config: Union[TrainingConfig, str, Dict]):
        if isinstance(config, str):
            config = self.load_config(config)
        elif isinstance(config, dict):
            config = TrainingConfig(**config)
        
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize models
        self.model = None
        self.teacher_model = None
        
        # Initialize logging
        if self.config.use_wandb:
            self.init_wandb()
    
    @staticmethod
    def load_config(config_path: str) -> TrainingConfig:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Flatten nested config
        flat_config = {}
        for section in config_dict.values():
            if isinstance(section, dict):
                flat_config.update(section)
        
        # Filter to only known TrainingConfig fields
        import dataclasses
        valid_fields = {f.name for f in dataclasses.fields(TrainingConfig)}
        flat_config = {k: v for k, v in flat_config.items() if k in valid_fields}
        
        return TrainingConfig(**flat_config)
    
    def init_wandb(self):
        """Initialize Weights & Biases logging."""
        wandb.init(
            project=self.config.project_name,
            name=self.config.run_name,
            config=vars(self.config)
        )
    
    def load_models(self):
        """Load base and teacher models."""
        logger.info(f"Loading base model: {self.config.base_model}")
        self.model = SentenceTransformer(self.config.base_model, token=HF_TOKEN)
        self.model.max_seq_length = self.config.max_seq_length
        
        logger.info(f"Loading teacher model: {self.config.teacher_model}")
        self.teacher_model = SentenceTransformer(self.config.teacher_model, token=HF_TOKEN)
        self.teacher_model.to(self.device)
    
    def create_loss_function(self) -> nn.Module:
        """Create the appropriate loss function based on config."""
        loss_type = self.config.loss_type.lower()
        
        if loss_type == "mnrl":
            base_loss = losses.MultipleNegativesRankingLoss(self.model)
        elif loss_type == "cosine":
            base_loss = losses.CosineSimilarityLoss(self.model)
        elif loss_type == "contrastive":
            base_loss = losses.ContrastiveLoss(self.model)
        elif loss_type == "mse":
            base_loss = losses.MSELoss(self.model)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
        
        # Wrap with Matryoshka loss if enabled
        if self.config.use_matryoshka:
            loss = losses.MatryoshkaLoss(
                model=self.model,
                loss=base_loss,
                matryoshka_dims=self.config.matryoshka_dims
            )
            logger.info(f"Using Matryoshka loss with dims: {self.config.matryoshka_dims}")
        else:
            loss = base_loss
        
        return loss
    
    def create_evaluator(self, eval_dataset: BanglaEmbeddingDataset):
        """Create evaluation objects."""
        # Prepare evaluation data
        sentences1 = []
        sentences2 = []
        scores = []
        
        for example in eval_dataset.examples:
            if len(example.texts) >= 2 and example.label is not None:
                sentences1.append(example.texts[0])
                sentences2.append(example.texts[1])
                scores.append(example.label)
        
        if sentences1:
            evaluator = evaluation.EmbeddingSimilarityEvaluator(
                sentences1=sentences1,
                sentences2=sentences2,
                scores=scores,
                name="bangla-sts-eval"
            )
            return evaluator
        
        return None
    
    def train_with_sentence_transformers(self):
        """
        Train using the sentence-transformers library's built-in trainer.
        """
        logger.info("Starting training with sentence-transformers trainer...")
        
        # Load models
        self.load_models()
        
        # Load datasets
        train_dataset = BanglaEmbeddingDataset(
            self.config.train_data_path,
            max_samples=None
        )
        
        eval_dataset = BanglaEmbeddingDataset(
            self.config.eval_data_path,
            max_samples=5000
        )
        
        # Create loss
        train_loss = self.create_loss_function()
        
        # Create evaluator
        evaluator = self.create_evaluator(eval_dataset)
        
        # Training arguments
        training_args = SentenceTransformerTrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            warmup_ratio=self.config.warmup_ratio,
            weight_decay=self.config.weight_decay,
            fp16=self.config.fp16,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            eval_steps=self.config.eval_steps,
            save_steps=self.config.save_steps,
            logging_steps=100,
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
        )
        
        # Create trainer
        trainer = SentenceTransformerTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            loss=train_loss,
            evaluator=evaluator,
        )
        
        # Train
        trainer.train()
        
        # Save final model
        final_path = Path(self.config.output_dir) / "final"
        self.model.save(str(final_path))
        logger.info(f"Saved final model to {final_path}")
        
        return self.model
    
    def train_with_distillation(self):
        """
        Train using cross-lingual knowledge distillation. 
        Teacher model provides English embeddings, student learns Bangla. 
        """
        logger.info("Starting cross-lingual distillation training...")
        
        # Load models
        self.load_models()
        
        # Load parallel data
        distillation_dataset = DistillationDataset(
            self.config.parallel_data_path,
            self.teacher_model
        )
        
        # Create dataloader
        train_dataloader = create_dataloader(
            distillation_dataset,
            batch_size=self.config.batch_size
        )
        
        # MSE Loss for distillation
        mse_loss = nn.MSELoss()
        
        # Optimizer
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Learning rate scheduler
        total_steps = len(train_dataloader) * self.config.num_epochs
        warmup_steps = int(total_steps * self.config.warmup_ratio)
        
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_steps
        )
        
        # Training loop
        self.model.to(self.device)
        self.model.train()
        
        global_step = 0
        best_loss = float('inf')
        
        for epoch in range(self.config.num_epochs):
            epoch_loss = 0.0
            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")
            
            for batch in progress_bar:
                optimizer.zero_grad()
                
                # Get student embeddings for Bangla texts
                student_embeddings = self.model.encode(
                    batch['bangla_text'],
                    convert_to_tensor=True
                )
                
                # Teacher embeddings (pre-computed)
                teacher_embeddings = batch['teacher_embedding']. to(self.device)
                
                # Compute MSE loss
                loss = mse_loss(student_embeddings, teacher_embeddings)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                scheduler.step()
                
                epoch_loss += loss.item()
                global_step += 1
                
                progress_bar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'lr': f"{scheduler.get_last_lr()[0]:.2e}"
                })
                
                # Logging
                if self.config.use_wandb and global_step % 100 == 0:
                    wandb.log({
                        'train_loss': loss.item(),
                        'learning_rate': scheduler.get_last_lr()[0],
                        'epoch': epoch + 1,
                        'global_step': global_step
                    })
                
                # Save checkpoint
                if global_step % self.config.save_steps == 0:
                    checkpoint_path = Path(self.config.output_dir) / f"checkpoint-{global_step}"
                    self.model.save(str(checkpoint_path))
            
            # End of epoch
            avg_epoch_loss = epoch_loss / len(train_dataloader)
            logger.info(f"Epoch {epoch + 1} - Average Loss: {avg_epoch_loss:.4f}")
            
            # Save best model
            if avg_epoch_loss < best_loss: 
                best_loss = avg_epoch_loss
                best_path = Path(self.config.output_dir) / "best"
                self.model.save(str(best_path))
                logger.info(f"Saved best model with loss:  {best_loss:.4f}")
        
        # Save final model
        final_path = Path(self.config.output_dir) / "final"
        self.model.save(str(final_path))
        
        if self.config.use_wandb:
            wandb.finish()
        
        return self.model
    
    def train_multi_stage(self):
        """
        Multi-stage training pipeline: 
        1. Cross-lingual distillation
        2. Fine-tuning on Bangla task data
        3. Hard negative mining refinement
        """
        logger.info("Starting multi-stage training pipeline...")
        
        # Stage 1: Cross-lingual distillation
        logger.info("=== Stage 1: Cross-lingual Distillation ===")
        self.config.num_epochs = 5
        self.train_with_distillation()
        
        # Stage 2: Fine-tuning on Bangla data
        logger.info("=== Stage 2: Fine-tuning on Bangla Data ===")
        self.config.num_epochs = 5
        self.config.learning_rate = 1e-5  # Lower LR for fine-tuning
        self.train_with_sentence_transformers()
        
        # Stage 3: Hard negative mining refinement
        logger.info("=== Stage 3: Hard Negative Mining ===")
        self.train_with_hard_negatives()
        
        return self.model
    
    def train_with_hard_negatives(self):
        """
        Fine-tune with hard negative mining.
        """
        logger.info("Training with hard negative mining...")
        
        # Create hard negative dataset
        hn_dataset = HardNegativeMiningDataset(
            self.config.train_data_path,
            self.model,
            num_hard_negatives=5
        )
        
        # Use Multiple Negatives Ranking Loss
        train_loss = losses.MultipleNegativesRankingLoss(self.model)
        
        # Create dataloader
        train_dataloader = create_dataloader(
            hn_dataset,
            batch_size=self.config.batch_size
        )
        
        # Train for a few epochs
        self.model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=3,
            warmup_steps=100,
            output_path=str(Path(self.config.output_dir) / "hard-negative-refined")
        )
        
        return self.model


def train_bangla_embedding_model(config_path: str, training_mode: str = "standard"):
    """
    Main training function. 
    
    Args:
        config_path: Path to training config YAML file
        training_mode: One of "standard", "distillation", or "multi_stage"
    """
    trainer = BanglaEmbeddingTrainer(config_path)
    
    if training_mode == "standard":
        return trainer.train_with_sentence_transformers()
    elif training_mode == "distillation": 
        return trainer.train_with_distillation()
    elif training_mode == "multi_stage": 
        return trainer.train_multi_stage()
    else:
        raise ValueError(f"Unknown training mode: {training_mode}")


if __name__ == "__main__": 
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/training_config.yaml")
    parser.add_argument("--mode", type=str, default="standard",
                       choices=["standard", "distillation", "multi_stage"])
    args = parser.parse_args()
    
    train_bangla_embedding_model(args.config, args.mode)