"""
Utility functions for Bangla embedding model training. 
"""

import os
import json
import random
from typing import List, Dict, Optional
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def count_parameters(model) -> Dict[str, int]:
    """Count model parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "total":  total,
        "trainable":  trainable,
        "frozen": total - trainable
    }


def save_json(data: Dict, filepath: str):
    """Save dictionary to JSON file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(filepath: str) -> Dict:
    """Load dictionary from JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def chunk_list(lst: List, chunk_size: int) -> List[List]:
    """Split list into chunks."""
    return [lst[i: i + chunk_size] for i in range(0, len(lst), chunk_size)]


def compute_similarity_matrix(
    embeddings1: torch.Tensor,
    embeddings2: torch.Tensor
) -> torch.Tensor:
    """Compute cosine similarity matrix between two sets of embeddings."""
    # Normalize embeddings
    embeddings1 = torch.nn.functional.normalize(embeddings1, p=2, dim=1)
    embeddings2 = torch.nn.functional.normalize(embeddings2, p=2, dim=1)
    
    # Compute similarity
    return torch.mm(embeddings1, embeddings2.t())


class EarlyStopping:
    """Early stopping handler."""
    
    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.0,
        mode: str = "min"
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == "min":
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


def format_time(seconds: float) -> str:
    """Format seconds to human-readable string."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"