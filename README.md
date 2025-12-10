# Bangla Embedding Model

A high-quality sentence embedding model for Bengali (Bangla) language, trained using cross-lingual knowledge distillation and optimized for the MTEB (Massive Text Embedding Benchmark) leaderboard.

## Features

- 🚀 **Multi-stage Training Pipeline**: Cross-lingual distillation → Fine-tuning → Hard negative mining
- 🎯 **Matryoshka Representation Learning**: Flexible embedding dimensions (768, 512, 256, 128, 64)
- 📊 **MTEB Benchmark Evaluation**: Built-in evaluation on Bengali MTEB tasks
- 🔄 **Multiple Loss Functions**: MNRL, Cosine Similarity, Contrastive, MSE
- 📈 **Weights & Biases Integration**: Full experiment tracking support

## Project Structure

```
bangla-embedding-model/
├── configs/
│   └── training_config.yaml    # Training configuration
├── data/
│   ├── raw/                    # Raw downloaded data
│   ├── processed/              # Processed training data
│   └── parallel/               # Parallel corpus data
├── scripts/
│   ├── download_data.sh        # Data download script
│   ├── train.py                # Main training script
│   └── evaluate_mteb.py        # MTEB evaluation script
├── src/
│   ├── __init__.py
│   ├── data_preparation.py     # Data loading and preprocessing
│   ├── dataset.py              # Custom PyTorch datasets
│   ├── evaluate.py             # Evaluation utilities
│   ├── trainer.py              # Training logic
│   └── utils.py                # Utility functions
├── .env.example                # Environment variables template
├── requirements.txt            # Python dependencies
└── README.md
```

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/bangla-embedding-model.git
cd bangla-embedding-model
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

```bash
# Copy the example .env file
cp .env.example .env

# Edit .env and add your tokens
nano .env  # or use your preferred editor
```

Add your Hugging Face token to the `.env` file:
```
HF_TOKEN=your_huggingface_token_here
WANDB_API_KEY=your_wandb_api_key_here  # Optional
```

## Quick Start

### Testing the Installation

```bash
# Test that all imports work
python -c "from src import BanglaEmbeddingTrainer, TrainingConfig; print('✓ All imports successful!')"
```

### Running a Quick Test

```python
from sentence_transformers import SentenceTransformer

# Load a pre-trained multilingual model (base model)
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")

# Test with Bangla sentences
sentences = [
    "বাংলাদেশ একটি সুন্দর দেশ।",
    "এটি একটি সুন্দর দিন।",
    "আজ আবহাওয়া ভালো।"
]

# Generate embeddings
embeddings = model.encode(sentences)
print(f"Generated {len(embeddings)} embeddings with dimension {embeddings[0].shape[0]}")
```

## Training

### 1. Prepare Training Data

```bash
# Download and prepare datasets
bash scripts/download_data.sh

# Or using Python
python -c "from src.data_preparation import prepare_training_data; prepare_training_data('./data/processed')"
```

### 2. Configure Training

Edit `configs/training_config.yaml` to customize:

```yaml
model:
  base_model: "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
  max_seq_length: 256

training:
  num_epochs: 10
  batch_size: 32
  learning_rate: 2e-5

loss:
  primary: "mnrl"
  use_matryoshka: true
  matryoshka_dims: [768, 512, 256, 128, 64]
```

### 3. Run Training

```bash
# Standard training
python scripts/train.py --config configs/training_config.yaml --mode standard

# Multi-stage training (recommended)
python scripts/train.py --config configs/training_config.yaml --mode multi_stage

# With data preparation
python scripts/train.py --config configs/training_config.yaml --mode multi_stage --prepare_data

# Train and evaluate
python scripts/train.py --config configs/training_config.yaml --mode multi_stage --evaluate

# Train and push to Hugging Face Hub
python scripts/train.py --config configs/training_config.yaml --mode multi_stage --push_to_hub --hub_model_id your-username/bangla-embedding
```

### Training Modes

| Mode | Description |
|------|-------------|
| `standard` | Standard sentence-transformers training |
| `distillation` | Cross-lingual knowledge distillation only |
| `multi_stage` | Full pipeline: distillation → fine-tuning → hard negative mining |

## Evaluation

### Run MTEB Evaluation

```bash
# Evaluate on Bengali MTEB tasks
python scripts/evaluate_mteb.py --model_path ./outputs/bangla-embedding-model/final

# Evaluate a Hugging Face model
python scripts/evaluate_mteb.py --model_path your-username/bangla-embedding

# Generate MTEB leaderboard submission
python scripts/evaluate_mteb.py --model_path ./outputs/bangla-embedding-model/final --generate_submission
```

### Programmatic Evaluation

```python
from src.evaluate import BanglaEmbeddingEvaluator, run_full_evaluation

# Quick evaluation
evaluator = BanglaEmbeddingEvaluator("./outputs/bangla-embedding-model/final")
results = evaluator.evaluate_on_bengali()
print(results)

# Full evaluation with submission files
results = run_full_evaluation(
    model_path="./outputs/bangla-embedding-model/final",
    output_dir="./evaluation_results"
)
```

## Using the Trained Model

### Load from Local Path

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("./outputs/bangla-embedding-model/final")
embeddings = model.encode(["বাংলা টেক্সট"])
```

### Load from Hugging Face Hub

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("your-username/bangla-embedding")
embeddings = model.encode(["বাংলা টেক্সট"])
```

### Semantic Similarity

```python
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("./outputs/bangla-embedding-model/final")

sentences = [
    "বাংলাদেশ দক্ষিণ এশিয়ার একটি দেশ।",
    "বাংলাদেশ এশিয়া মহাদেশে অবস্থিত।",
    "আজ আবহাওয়া খুব ভালো।"
]

embeddings = model.encode(sentences)

# Compute similarity
similarity = util.cos_sim(embeddings[0], embeddings[1])
print(f"Similarity between sentence 1 and 2: {similarity.item():.4f}")

similarity = util.cos_sim(embeddings[0], embeddings[2])
print(f"Similarity between sentence 1 and 3: {similarity.item():.4f}")
```

### Using Matryoshka Dimensions

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("./outputs/bangla-embedding-model/final")

# Full dimension (768)
embeddings_768 = model.encode(["বাংলা টেক্সট"])

# Truncate to smaller dimensions for efficiency
embeddings_256 = embeddings_768[:, :256]  # 256-dim
embeddings_128 = embeddings_768[:, :128]  # 128-dim
embeddings_64 = embeddings_768[:, :64]    # 64-dim
```

## Pushing to Hugging Face Hub

### Method 1: Using the Training Script

```bash
# Make sure .env has your HF_TOKEN
python scripts/train.py --config configs/training_config.yaml --push_to_hub --hub_model_id your-username/model-name
```

### Method 2: Manual Push

```python
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv

load_dotenv()

model = SentenceTransformer("./outputs/bangla-embedding-model/final")
model.push_to_hub(
    "your-username/bangla-embedding",
    token=os.getenv("HF_TOKEN")
)
```

### Method 3: Using Hugging Face CLI

```bash
# Login first
huggingface-cli login

# Then push
python -c "
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('./outputs/bangla-embedding-model/final')
model.push_to_hub('your-username/bangla-embedding')
"
```

## Configuration Reference

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `HF_TOKEN` | Hugging Face API token for model upload | For push_to_hub |
| `HUGGINGFACE_TOKEN` | Alternative HF token variable | For push_to_hub |
| `WANDB_API_KEY` | Weights & Biases API key | Optional |
| `WANDB_PROJECT` | W&B project name | Optional |

### Training Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `base_model` | `paraphrase-multilingual-mpnet-base-v2` | Base model for fine-tuning |
| `teacher_model` | `all-mpnet-base-v2` | Teacher model for distillation |
| `max_seq_length` | 256 | Maximum sequence length |
| `num_epochs` | 10 | Number of training epochs |
| `batch_size` | 32 | Training batch size |
| `learning_rate` | 2e-5 | Learning rate |
| `loss_type` | `mnrl` | Loss function type |
| `use_matryoshka` | true | Enable Matryoshka learning |

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce `batch_size` in config
   - Enable `gradient_accumulation_steps`
   - Set `fp16: true`

2. **Missing Data Files**
   - Run `bash scripts/download_data.sh` first
   - Or use `--prepare_data` flag

3. **Import Errors**
   - Ensure you're in the project root directory
   - Check that all dependencies are installed

4. **HF Token Issues**
   - Verify token in `.env` file
   - Try `huggingface-cli login` as alternative

### Debug Mode

```bash
# Run with verbose logging
PYTHONPATH=. python -c "
import logging
logging.basicConfig(level=logging.DEBUG)
from src.trainer import BanglaEmbeddingTrainer
trainer = BanglaEmbeddingTrainer('configs/training_config.yaml')
print('Config loaded successfully!')
print(trainer.config)
"
```

## Citation

If you use this model in your research, please cite:

```bibtex
@misc{bangla-embedding-model,
  author = {Your Name},
  title = {Bangla Embedding Model},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/yourusername/bangla-embedding-model}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Sentence Transformers](https://www.sbert.net/)
- [MTEB Benchmark](https://github.com/embeddings-benchmark/mteb)
- [Hugging Face](https://huggingface.co/)
