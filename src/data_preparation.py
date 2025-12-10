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


def normalize_bangla_text(text: str, use_normalizer: bool = False) -> str:
    """Normalize Bangla text.
    
    Args:
        text: Input text to normalize
        use_normalizer: If True, use bnunicodenormalizer (slower but thorough).
                       If False, just do basic whitespace normalization (fast).
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Skip expensive word-by-word normalization for speed
    # The BanglaParaphrase dataset is already clean
    if not use_normalizer:
        return text
    
    # Apply Bangla unicode normalization word-by-word (slow)
    normalized_words = []
    for word in text.split():
        try:
            result = bn_normalizer(word)
            if result and result.get('normalized'):
                normalized_words.append(result['normalized'])
            else:
                normalized_words.append(word)
        except Exception:
            normalized_words.append(word)
    
    return ' '.join(normalized_words)


def clean_text(text: str, lang: str = "bn", thorough: bool = False) -> str:
    """Clean and normalize text based on language.
    
    Args:
        text: Input text
        lang: Language code ('bn' for Bangla)
        thorough: If True, use thorough normalization (slower)
    """
    if not text: 
        return ""
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Language-specific cleaning
    if lang == "bn":
        text = normalize_bangla_text(text, use_normalizer=thorough)
    else:
        text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def load_parallel_corpus(data_dir: str) -> List[Dict]:
    """
    Load English-Bangla parallel corpus for knowledge distillation.
    Note: OPUS datasets may not be directly available. This function
    will try to load from local files if available.
    """
    parallel_data = []
    data_path = Path(data_dir)
    
    # First, try to load from local parallel data directory
    local_parallel_file = data_path / "en_bn_parallel.json"
    if local_parallel_file.exists():
        print(f"Loading parallel data from {local_parallel_file}")
        with open(local_parallel_file, 'r', encoding='utf-8') as f:
            parallel_data = json.load(f)
        print(f"Loaded {len(parallel_data)} parallel sentence pairs from local file")
        return parallel_data
    
    print("No local parallel corpus found. Skipping parallel data loading.")
    print("To use parallel data, place en_bn_parallel.json in data/parallel/")
    return parallel_data


def load_paraphrase_data(data_dir: str, max_samples: Optional[int] = None) -> List[Dict]:
    """Load Bangla paraphrase data from the downloaded BanglaParaphrase dataset.
    
    Args:
        data_dir: Data directory path
        max_samples: Maximum number of samples to load (None for all)
    """
    paraphrase_data = []
    
    try:
        # Load from the downloaded dataset on disk
        from datasets import load_from_disk
        
        dataset_path = Path(data_dir).parent / "raw" / "paraphrase_bn"
        if not dataset_path.exists():
            # Try alternate path
            dataset_path = Path("data/raw/paraphrase_bn")
        
        print(f"Loading paraphrase data from {dataset_path}")
        dataset = load_from_disk(str(dataset_path))
        
        samples_loaded = 0
        for split in ['train', 'validation', 'test']:
            if split in dataset:
                split_data = dataset[split]
                # Limit samples if max_samples is set
                if max_samples and samples_loaded >= max_samples:
                    break
                
                remaining = max_samples - samples_loaded if max_samples else len(split_data)
                items_to_process = min(len(split_data), remaining)
                
                # Fast batch processing - directly access data
                for i in tqdm(range(items_to_process), desc=f"Processing paraphrase {split}"):
                    item = split_data[i]
                    # BanglaParaphrase has 'source' and 'target' fields
                    sent1 = clean_text(item['source'], 'bn')
                    sent2 = clean_text(item['target'], 'bn')
                    
                    # All pairs in BanglaParaphrase are valid paraphrases (label=1.0)
                    if len(sent1) > 5 and len(sent2) > 5:
                        paraphrase_data.append({
                            "sentence1": sent1,
                            "sentence2": sent2,
                            "label": 1.0,  # All pairs are paraphrases
                            "task": "paraphrase"
                        })
                        samples_loaded += 1
                        
                        if max_samples and samples_loaded >= max_samples:
                            break
                        
    except Exception as e:
        print(f"Error loading paraphrase data: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"Loaded {len(paraphrase_data)} paraphrase pairs")
    return paraphrase_data


def load_nli_data(data_dir: str) -> List[Dict]:
    """Load NLI data for Bangla (from local xnli_bn or XNLI)."""
    nli_data = []
    
    # First try to load from local downloaded data
    try:
        from datasets import load_from_disk
        
        local_nli_path = Path("data/raw/xnli_bn")
        if local_nli_path.exists():
            print(f"Loading NLI data from {local_nli_path}")
            dataset = load_from_disk(str(local_nli_path))
            
            label_map = {0: "entailment", 1: "neutral", 2: "contradiction"}
            
            for split in ['train', 'validation', 'test']:
                if split in dataset:
                    for item in tqdm(dataset[split], desc=f"Processing NLI {split}"):
                        premise = clean_text(item.get('premise', item.get('sentence1', '')), 'bn')
                        hypothesis = clean_text(item.get('hypothesis', item.get('sentence2', '')), 'bn')
                        label_val = item.get('label', item.get('gold_label', 1))
                        label = label_map.get(label_val, 'neutral') if isinstance(label_val, int) else label_val
                        
                        if len(premise) > 5 and len(hypothesis) > 5:
                            nli_data.append({
                                "premise": premise,
                                "hypothesis": hypothesis,
                                "label": label,
                                "task": "nli"
                            })
            
            print(f"Loaded {len(nli_data)} NLI pairs from local data")
            return nli_data
    except Exception as e:
        print(f"Could not load local NLI data: {e}")
    
    # Fallback: try loading from Hugging Face
    try:
        print("Trying to load XNLI from Hugging Face...")
        dataset = load_dataset("facebook/xnli", "bn")
        
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
        print(f"NLI data not available: {e}")
        print("Skipping NLI data. Training will proceed with paraphrase data only.")
    
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
    include_nli: bool = True,
    include_synthetic: bool = False,
    max_samples: Optional[int] = None
) -> Tuple[str, str]: 
    """
    Main function to prepare all training data.
    Returns paths to train and eval data files.
    
    Args:
        output_dir: Directory to save processed data
        include_parallel: Include parallel corpus data
        include_paraphrase: Include paraphrase data
        include_nli: Include NLI data
        include_synthetic: Include synthetic data
        max_samples: Maximum total samples (None for all). Use for quick testing.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    all_data = []
    
    if max_samples:
        print(f"\n*** FAST MODE: Limiting to {max_samples} samples ***\n")
    
    # Load different data sources
    if include_parallel:
        parallel_data = load_parallel_corpus(str(output_path / "parallel"))
        all_data.extend([{
            "texts": [item["english"], item["bangla"]],
            "label": 1.0,
            "task": "parallel"
        } for item in parallel_data])
    
    if include_paraphrase:
        # Pass max_samples to limit data loading
        paraphrase_data = load_paraphrase_data(str(output_path), max_samples=max_samples)
        all_data.extend([{
            "texts": [item["sentence1"], item["sentence2"]],
            "label": item["label"],
            "task": "paraphrase"
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