# Named Entity Recognition (NER) Project

A fine-tuned BERT-based Named Entity Recognition model for identifying person names, organizations, locations, and miscellaneous entities.

## Overview

This project trains a token classification model using the Hugging Face Transformers library. The model achieves strong performance on NER tasks with an overall F1-score of 0.91.

### Model Performance

| Entity Type | Precision | Recall | F1-Score |
|-------------|-----------|--------|----------|
| LOC         | 0.92      | 0.93   | 0.93     |
| MISC        | 0.79      | 0.84   | 0.82     |
| ORG         | 0.88      | 0.90   | 0.89     |
| PER         | 0.97      | 0.96   | 0.96     |
| **Overall** | **0.91**  | **0.92** | **0.91** |

## Project Structure

```
├── train_ner.py                 # Training script
├── use_model_prompt.py          # Inference script
├── train_requirements.txt       # Python dependencies
├── train.txt                    # Training data (CoNLL format)
├── test.txt                     # Test data (CoNLL format)
├── valid.txt                    # Validation data (CoNLL format)
└── ner-output/                  # Model outputs (excluded from Git)
    ├── config.json
    ├── model.safetensors
    ├── tokenizer files
    └── checkpoints/
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Exjobb.git
cd Exjobb
```

2. Install dependencies:
```bash
pip install -r train_requirements.txt
```

## Usage

### Training

Train the NER model on your CoNLL-formatted data:

```bash
python train_ner.py \
    --train_file train.txt \
    --test_file test.txt \
    --valid_file valid.txt \
    --output_dir ner-output \
    --num_train_epochs 3 \
    --per_device_train_batch_size 16
```

### Inference

Run the trained model on new text:

```bash
# Single prompt
python use_model_prompt.py --model_dir ner-output --prompt "John lives in New York City."

# From file
python use_model_prompt.py --model_dir ner-output --input_file prompts.txt --output_file results.json
```

## Data Format

The training data follows the CoNLL-2003 format:
```
John POS CHUNK B-PER
lives POS CHUNK O
in POS CHUNK O
New POS CHUNK B-LOC
York POS CHUNK I-LOC
```

## Requirements

- Python 3.7+
- PyTorch
- Transformers >= 4.30.0
- datasets
- evaluate
- seqeval

## Model Storage

⚠️ **Note**: The trained model files (`.safetensors`, `.pt`) are large (>100MB) and are excluded from Git via `.gitignore`. 

To share or version control models:
- Use [Hugging Face Hub](https://huggingface.co/models) for model hosting
- Or use [Git LFS](https://git-lfs.github.com/) for large file storage

## License

[Add your license here]

## Author

[Add your name here]
