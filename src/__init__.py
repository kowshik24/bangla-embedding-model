"""
Bangla Embedding Model - Training pipeline for MTEB benchmark.
"""

from .data_preparation import (
    normalize_bangla_text,
    clean_text,
    load_parallel_corpus,
    load_paraphrase_data,
    load_nli_data,
    create_triplets_from_nli,
    prepare_training_data
)

from .dataset import (
    BanglaEmbeddingExample,
    BanglaEmbeddingDataset,
    HardNegativeMiningDataset,
    DistillationDataset,
    create_dataloader
)

from .trainer import (
    TrainingConfig,
    BanglaEmbeddingTrainer,
    train_bangla_embedding_model
)

from .evaluate import (
    BanglaEmbeddingEvaluator,
    run_full_evaluation
)

from .utils import (
    set_seed,
    get_device,
    count_parameters,
    save_json,
    load_json,
    chunk_list,
    compute_similarity_matrix,
    EarlyStopping,
    format_time
)

__version__ = "0.1.0"
__all__ = [
    # Data preparation
    "normalize_bangla_text",
    "clean_text",
    "load_parallel_corpus",
    "load_paraphrase_data",
    "load_nli_data",
    "create_triplets_from_nli",
    "prepare_training_data",
    # Dataset
    "BanglaEmbeddingExample",
    "BanglaEmbeddingDataset",
    "HardNegativeMiningDataset",
    "DistillationDataset",
    "create_dataloader",
    # Trainer
    "TrainingConfig",
    "BanglaEmbeddingTrainer",
    "train_bangla_embedding_model",
    # Evaluate
    "BanglaEmbeddingEvaluator",
    "run_full_evaluation",
    # Utils
    "set_seed",
    "get_device",
    "count_parameters",
    "save_json",
    "load_json",
    "chunk_list",
    "compute_similarity_matrix",
    "EarlyStopping",
    "format_time",
]