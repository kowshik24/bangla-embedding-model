"""
Data preparation module for Bangla embedding model training.
Handles data loading, cleaning, and preprocessing. 
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset, Dataset, concatenate_datasets
from bnunicodenormalizer import Normalizer

# Initialize Bangla normalizer
bn_normalizer = Normalizer()


def normalize_bangla_text(text:  str) -> str:
    """Normalize Bangla text using bnunicodenormalizer."""
    if not text or not isinstance(text, str):
        return ""
    
    # Apply Bangla unicode normalization
    normalized = bn_normalizer(text)
    if normalized and normalized['normalized']:
        text = normalized['normalized']
    
    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def clean_text(text: str, lang: str = "bn") -> str:
    """Clean and normalize text based on language."""
    if not text: 
        return ""
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Language-specific cleaning
    if lang == "bn":
        text = normalize_bangla_text(text)
    else:
        text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def load_parallel_corpus(data_dir: str) -> List[Dict]:
    """
    Load English-Bangla parallel corpus for knowledge distillation.
    """
    parallel_data = []
    data_path = Path(data_dir)
    
    # Try loading from various sources
    sources = [
        ("opus", "CCAligned", "en", "bn"),
        ("opus", "WikiMatrix", "en", "bn"),
    ]
    
    for source_type, name, src_lang, tgt_lang in sources:
        try:
            print(f"Loading {name} parallel corpus...")
            dataset = load_dataset(
                source_type, 
                name, 
                lang1=src_lang, 
                lang2=tgt_lang,
                trust_remote_code=True
            )
            
            for item in tqdm(dataset['train'], desc=f"Processing {name}"):
                en_text = clean_text(item['translation']['en'], 'en')
                bn_text = clean_text(item['translation']['bn'], 'bn')
                
                if len(en_text) > 10 and len(bn_text) > 10:
                    parallel_data.append({
                        "english": en_text,
                        "bangla": bn_text,
                        "source": name
                    })
                    
        except Exception as e:
            print(f"Could not load {name}:  {e}")
            continue
    
    print(f"Loaded {len(parallel_data)} parallel sentence pairs")
    return parallel_data


def load_paraphrase_data(data_dir: str) -> List[Dict]:
    """Load Bangla paraphrase identification data."""
    paraphrase_data = []
    
    try:
        dataset = load_dataset(
            "csebuetnlp/paraphrase_identification_bn",
            trust_remote_code=True
        )
        
        for split in ['train', 'validation']: 
            if split in dataset:
                for item in tqdm(dataset[split], desc=f"Processing paraphrase {split}"):
                    sent1 = clean_text(item['sentence1'], 'bn')
                    sent2 = clean_text(item['sentence2'], 'bn')
                    label = item['label']
                    
                    if len(sent1) > 5 and len(sent2) > 5:
                        paraphrase_data.append({
                            "sentence1": sent1,
                            "sentence2": sent2,
                            "label":  float(label),
                            "task": "paraphrase"
                        })
                        
    except Exception as e: 
        print(f"Error loading paraphrase data: {e}")
    
    print(f"Loaded {len(paraphrase_data)} paraphrase pairs")
    return paraphrase_data


def load_nli_data(data_dir:  str) -> List[Dict]:
    """Load NLI data for Bangla (from XNLI or similar)."""
    nli_data = []
    
    try:
        # Try loading XNLI Bengali subset
        dataset = load_dataset("xnli", "bn", trust_remote_code=True)
        
        label_map = {0: "entailment", 1: "neutral", 2: "contradiction"}
        
        for split in ['train', 'validation']:
            if split in dataset:
                for item in tqdm(dataset[split], desc=f"Processing NLI {split}"):
                    premise = clean_text(item['premise'], 'bn')
                    hypothesis = clean_text(item['hypothesis'], 'bn')
                    label = label_map.get(item['label'], 'neutral')
                    
                    if len(premise) > 5 and len(hypothesis) > 5:
                        nli_data.append({
                            "premise": premise,
                            "hypothesis": hypothesis,
                            "label": label,
                            "task": "nli"
                        })
                        
    except Exception as e:
        print(f"Error loading NLI data: {e}")
    
    print(f"Loaded {len(nli_data)} NLI pairs")
    return nli_data


def create_triplets_from_nli(nli_data: List[Dict]) -> List[Dict]:
    """
    Create anchor-positive-negative triplets from NLI data.
    Entailment pairs become positives, contradiction pairs become negatives. 
    """
    triplets = []
    
    # Group by premise
    premise_groups = {}
    for item in nli_data:
        premise = item['premise']
        if premise not in premise_groups:
            premise_groups[premise] = {'entailment': [], 'contradiction': []}
        
        if item['label'] == 'entailment':
            premise_groups[premise]['entailment'].append(item['hypothesis'])
        elif item['label'] == 'contradiction':
            premise_groups[premise]['contradiction'].append(item['hypothesis'])
    
    # Create triplets
    for premise, hyps in premise_groups.items():
        if hyps['entailment'] and hyps['contradiction']:
            for pos in hyps['entailment']:
                for neg in hyps['contradiction']:
                    triplets.append({
                        "anchor": premise,
                        "positive": pos,
                        "negative": neg,
                        "task": "triplet"
                    })
    
    print(f"Created {len(triplets)} triplets from NLI data")
    return triplets


def generate_synthetic_pairs(
    texts: List[str],
    model_name: str = "google/mt5-base"
) -> List[Dict]:
    """
    Generate synthetic paraphrase pairs using back-translation.
    """
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    
    synthetic_pairs = []
    
    try:
        # Load translation models
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        for text in tqdm(texts[: 1000], desc="Generating synthetic pairs"):  # Limit for demo
            # This is a simplified example - in practice, use proper translation models
            synthetic_pairs.append({
                "sentence1": text,
                "sentence2": text,  # Placeholder - replace with actual paraphrase
                "label": 1.0,
                "task": "synthetic"
            })
            
    except Exception as e:
        print(f"Error in synthetic generation: {e}")
    
    return synthetic_pairs


def prepare_training_data(
    output_dir: str,
    include_parallel: bool = True,
    include_paraphrase: bool = True,
    include_nli:  bool = True,
    include_synthetic: bool = False
) -> Tuple[str, str]: 
    """
    Main function to prepare all training data.
    Returns paths to train and eval data files.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    all_data = []
    
    # Load different data sources
    if include_parallel:
        parallel_data = load_parallel_corpus(str(output_path / "parallel"))
        all_data.extend([{
            "texts": [item["english"], item["bangla"]],
            "label": 1.0,
            "task":  "parallel"
        } for item in parallel_data])
    
    if include_paraphrase: 
        paraphrase_data = load_paraphrase_data(str(output_path))
        all_data.extend([{
            "texts": [item["sentence1"], item["sentence2"]],
            "label": item["label"],
            "task":  "paraphrase"
        } for item in paraphrase_data])
    
    if include_nli:
        nli_data = load_nli_data(str(output_path))
        triplets = create_triplets_from_nli(nli_data)
        all_data.extend([{
            "texts": [item["anchor"], item["positive"], item["negative"]],
            "label": None,
            "task": "triplet"
        } for item in triplets])
    
    # Shuffle and split
    import random
    random.shuffle(all_data)
    
    split_idx = int(len(all_data) * 0.95)
    train_data = all_data[:split_idx]
    eval_data = all_data[split_idx:]
    
    # Save to files
    train_path = output_path / "train.json"
    eval_path = output_path / "eval.json"
    
    with open(train_path, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    
    with open(eval_path, 'w', encoding='utf-8') as f:
        json.dump(eval_data, f, ensure_ascii=False, indent=2)
    
    print(f"Saved {len(train_data)} training samples to {train_path}")
    print(f"Saved {len(eval_data)} evaluation samples to {eval_path}")
    
    return str(train_path), str(eval_path)


if __name__ == "__main__": 
    prepare_training_data("./data/processed")