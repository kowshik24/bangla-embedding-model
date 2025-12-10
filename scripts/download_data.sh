#!/bin/bash

# Create directories
mkdir -p data/raw data/processed data/parallel

echo "Downloading Bangla datasets..."

# Download from Hugging Face
python -c "
from datasets import load_dataset, Dataset, DatasetDict
import requests
import zipfile
import json
import os
from io import BytesIO

# Download Bangla paraphrase data directly from the zip file
print('Downloading Bangla paraphrase dataset...')
url = 'https://huggingface.co/datasets/csebuetnlp/BanglaParaphrase/resolve/main/data/BanglaParaphrase.zip'
response = requests.get(url)
with zipfile.ZipFile(BytesIO(response.content)) as z:
    z.extractall('data/raw/temp_paraphrase')

# Load JSONL files into datasets
def load_jsonl(filepath):
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return Dataset.from_list(data)

dataset = DatasetDict({
    'train': load_jsonl('data/raw/temp_paraphrase/BanglaParaphrase/train.jsonl'),
    'validation': load_jsonl('data/raw/temp_paraphrase/BanglaParaphrase/validation.jsonl'),
    'test': load_jsonl('data/raw/temp_paraphrase/BanglaParaphrase/test.jsonl')
})
dataset.save_to_disk('data/raw/paraphrase_bn')

# Clean up temp files
import shutil
shutil.rmtree('data/raw/temp_paraphrase')
print(f'Loaded {len(dataset[\"train\"])} train, {len(dataset[\"validation\"])} validation, {len(dataset[\"test\"])} test examples')

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