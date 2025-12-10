"""
Custom dataset classes for Bangla embedding model training.
"""

import json
import random
from typing import List, Dict, Optional, Union
from dataclasses import dataclass
import torch
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import InputExample
from tqdm import tqdm


@dataclass
class BanglaEmbeddingExample:
    """Data class for training examples."""
    texts: List[str]
    label: Optional[float] = None
    task: str = "default"


class BanglaEmbeddingDataset(Dataset):
    """
    PyTorch Dataset for Bangla embedding training.
    Supports multiple training formats:  pairs, triplets, and labeled pairs.
    """
    
    def __init__(
        self,
        data_path: str,
        max_samples: Optional[int] = None,
        task_filter: Optional[List[str]] = None
    ):
        self.examples = []
        self.load_data(data_path, max_samples, task_filter)
    
    def load_data(
        self,
        data_path: str,
        max_samples: Optional[int],
        task_filter: Optional[List[str]]
    ):
        """Load and parse training data from JSON file."""
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for item in tqdm(data, desc="Loading dataset"):
            if task_filter and item.get('task') not in task_filter:
                continue
            
            example = BanglaEmbeddingExample(
                texts=item['texts'],
                label=item.get('label'),
                task=item.get('task', 'default')
            )
            self.examples.append(example)
            
            if max_samples and len(self.examples) >= max_samples:
                break
        
        print(f"Loaded {len(self.examples)} examples")
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> InputExample:
        example = self.examples[idx]
        return InputExample(
            texts=example.texts,
            label=example.label
        )


class HardNegativeMiningDataset(Dataset):
    """
    Dataset with hard negative mining support.
    Mines hard negatives from the corpus using the current model.
    """
    
    def __init__(
        self,
        data_path: str,
        model,
        num_hard_negatives: int = 5,
        refresh_every: int = 1000
    ):
        self.data_path = data_path
        self.model = model
        self.num_hard_negatives = num_hard_negatives
        self.refresh_every = refresh_every
        self.call_count = 0
        
        self.load_base_data()
        self.mine_hard_negatives()
    
    def load_base_data(self):
        """Load base training data."""
        with open(self.data_path, 'r', encoding='utf-8') as f:
            self.base_data = json.load(f)
        
        # Extract all unique sentences for negative mining
        self.corpus = []
        for item in self.base_data:
            self.corpus.extend(item['texts'])
        self.corpus = list(set(self.corpus))
    
    def mine_hard_negatives(self):
        """Mine hard negatives using current model embeddings."""
        print("Mining hard negatives...")
        
        # Encode corpus
        corpus_embeddings = self.model.encode(
            self.corpus,
            convert_to_tensor=True,
            show_progress_bar=True
        )
        
        self.examples = []
        
        for item in tqdm(self.base_data, desc="Creating examples with hard negatives"):
            if len(item['texts']) < 2:
                continue
            
            anchor = item['texts'][0]
            positive = item['texts'][1]
            
            # Find hard negatives
            anchor_embedding = self.model.encode(anchor, convert_to_tensor=True)
            
            # Compute similarities
            similarities = torch.nn.functional.cosine_similarity(
                anchor_embedding.unsqueeze(0),
                corpus_embeddings
            )
            
            # Get indices of hard negatives (high similarity but not the positive)
            sorted_indices = torch.argsort(similarities, descending=True)
            
            hard_negatives = []
            for idx in sorted_indices:
                neg_text = self.corpus[idx]
                if neg_text != anchor and neg_text != positive:
                    hard_negatives.append(neg_text)
                    if len(hard_negatives) >= self.num_hard_negatives:
                        break
            
            # Create training example
            self.examples.append({
                'anchor': anchor,
                'positive': positive,
                'negatives': hard_negatives
            })
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> InputExample:
        self.call_count += 1
        
        # Refresh hard negatives periodically
        if self.call_count % self.refresh_every == 0:
            self.mine_hard_negatives()
        
        example = self.examples[idx]
        
        # Randomly select one hard negative
        negative = random.choice(example['negatives']) if example['negatives'] else ""
        
        return InputExample(
            texts=[example['anchor'], example['positive'], negative]
        )


class DistillationDataset(Dataset):
    """
    Dataset for cross-lingual knowledge distillation. 
    Uses English-Bangla parallel pairs. 
    """
    
    def __init__(
        self,
        parallel_data_path: str,
        teacher_model,
        max_samples: Optional[int] = None
    ):
        self.teacher_model = teacher_model
        self.examples = []
        
        self.load_data(parallel_data_path, max_samples)
        self.compute_teacher_embeddings()
    
    def load_data(self, data_path: str, max_samples: Optional[int]):
        """Load parallel corpus data."""
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for item in data:
            if 'english' in item and 'bangla' in item: 
                self.examples.append({
                    'english': item['english'],
                    'bangla': item['bangla']
                })
                
                if max_samples and len(self.examples) >= max_samples:
                    break
    
    def compute_teacher_embeddings(self):
        """Pre-compute teacher embeddings for English sentences."""
        print("Computing teacher embeddings...")
        
        english_texts = [ex['english'] for ex in self.examples]
        
        self.teacher_embeddings = self.teacher_model.encode(
            english_texts,
            convert_to_tensor=True,
            show_progress_bar=True
        )
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict:
        return {
            'bangla_text': self.examples[idx]['bangla'],
            'teacher_embedding': self.teacher_embeddings[idx]
        }


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4
) -> DataLoader:
    """Create a DataLoader for the given dataset."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )