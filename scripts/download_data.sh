#!/bin/bash

# Create directories
mkdir -p data/raw data/processed data/parallel

echo "Downloading Bangla datasets..."

# Download from Hugging Face
python -c "
from datasets import load_dataset

# Download Bangla paraphrase data
print('Downloading Bangla paraphrase dataset...')
dataset = load_dataset('csebuetnlp/BanglaParaphrase')
dataset.save_to_disk('data/raw/paraphrase_bn')

# Download Bangla NLI data (if available)
print('Downloading Bangla NLI dataset...')
try:
    nli_dataset = load_dataset('csebuetnlp/xnli_bn')
    nli_dataset.save_to_disk('data/raw/xnli_bn')
except:
    print('NLI dataset not found, skipping...')

# Download STS benchmark data
print('Downloading STS data...')
sts_dataset = load_dataset('mteb/sts12-sts')
sts_dataset.save_to_disk('data/raw/sts')

print('Download complete!')
"

echo "Data download complete!"