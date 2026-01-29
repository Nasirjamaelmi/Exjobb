import os
import argparse
from collections import defaultdict
import logging

import numpy as np
import torch
import random
import json
from seqeval.metrics import classification_report

from datasets import Dataset, load_dataset

import evaluate
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
    AutoModelForMaskedLM,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



def read_conll(file_path):
    """Read a CoNLL-style file (word POS chunk NER) and return list of examples.

    Each example is a dict with keys: 'tokens' (list[str]) and 'ner_tags' (list[str]).
    """
    examples = []
    tokens = []
    tags = []
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                if tokens:
                    examples.append({"tokens": tokens, "ner_tags": tags})
                    tokens = []
                    tags = []
                continue
            # Typical CoNLL-2003: token POS chunk NER
            parts = line.split()
            if len(parts) == 0:
                continue
            # last column is the NER tag, first column is the token
            token = parts[0]
            tag = parts[-1]
            tokens.append(token)
            tags.append(tag)
    if tokens:
        examples.append({"tokens": tokens, "ner_tags": tags})
    return examples


def get_label_list(examples_list):
    labels = set()
    for ex in examples_list:
        labels.update(ex["ner_tags"])
    # Keep order stable
    labels = sorted(labels)
    return labels


def align_labels_with_tokens(labels, label_to_id, tokenized_inputs, example_labels):
    aligned_labels = []
    # tokenized_inputs is result of tokenizer(...) with is_split_into_words=True
    for i in range(len(tokenized_inputs['input_ids'])):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                # first token of the word
                label_ids.append(label_to_id[example_labels[i][word_idx]])
            else:
                # For sub-tokens, set to -100 to be ignored in loss
                label_ids.append(-100)
            previous_word_idx = word_idx
        aligned_labels.append(label_ids)
    return aligned_labels


def tokenize_and_align_labels(dataset, tokenizer, label_to_id, max_length=128):
    # dataset is a list/dict with keys 'tokens' and 'ner_tags'
    tokens = [ex['tokens'] for ex in dataset]
    ner_tags = [ex['ner_tags'] for ex in dataset]

    tokenized_inputs = tokenizer(
        tokens,
        is_split_into_words=True,
        truncation=True,
        padding=False,
        max_length=max_length,
        return_attention_mask=True,
    )

    # Align labels
    aligned_labels = align_labels_with_tokens(ner_tags, label_to_id, tokenized_inputs, ner_tags)
    tokenized_inputs['labels'] = aligned_labels
    return tokenized_inputs


def compute_metrics(p, id_to_label):
    metric = evaluate.load("seqeval")
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = []
    true_labels = []

    for pred_row, label_row in zip(predictions, labels):
        pred_tags = []
        true_tags = []
        for p_id, l_id in zip(pred_row, label_row):
            if l_id == -100:
                continue
            pred_tags.append(id_to_label[p_id])
            true_tags.append(id_to_label[l_id])
        true_predictions.append(pred_tags)
        true_labels.append(true_tags)

    results = metric.compute(predictions=true_predictions, references=true_labels)
    overall = {}
    if 'overall_precision' in results:
        overall['precision'] = results['overall_precision']
        overall['recall'] = results['overall_recall']
        overall['f1'] = results['overall_f1']
    return overall


def main():
    parser = argparse.ArgumentParser(description="Train NER on local CoNLL-2003 files")
    parser.add_argument("--model_name", default="google-bert/bert-base-multilingual-cased", help="Pretrained model name or path (e.g. google/bert-base-multilingual-cased or xlm-roberta-base)")
    parser.add_argument("--train_file", default="train.txt", help="Path to train file")
    parser.add_argument("--validation_file", default="valid.txt", help="Path to validation file")
    parser.add_argument("--output_dir", default="ner-output", help="Model output directory")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--fp16", action="store_true", help="Enable mixed precision (fp16) when training on GPU")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps to effectively increase batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--test_file", default="test.txt", help="Path to test file to evaluate after training")
    args = parser.parse_args()

    logger.info("Reading datasets")
    train_examples = read_conll(args.train_file)
    val_examples = read_conll(args.validation_file)

    labels = get_label_list(train_examples + val_examples)
    label_to_id = {l: i for i, l in enumerate(labels)}
    id_to_label = {i: l for l, i in label_to_id.items()}

    logger.info(f"Labels ({len(labels)}): {labels}")

    # Set seeds for reproducibility
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # GPU / CUDA info
    cuda_available = torch.cuda.is_available()
    logger.info(f"torch.cuda.is_available={cuda_available}")
    if cuda_available:
        try:
            logger.info(f"GPU name: {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA device count: {torch.cuda.device_count()}")
        except Exception:
            logger.info("Could not query CUDA device name")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    logger.info("Tokenizing and aligning labels (train)")
    train_tokenized = tokenize_and_align_labels(train_examples, tokenizer, label_to_id, max_length=args.max_length)
    logger.info("Tokenizing and aligning labels (validation)")
    val_tokenized = tokenize_and_align_labels(val_examples, tokenizer, label_to_id, max_length=args.max_length)

    # Build Hugging Face Datasets
    train_dataset = Dataset.from_dict(train_tokenized)
    val_dataset = Dataset.from_dict(val_tokenized)

    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name, num_labels=len(labels)
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        logging_steps=50,
        weight_decay=0.01,
        fp16=args.fp16,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        save_total_limit=3,
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=lambda p: compute_metrics(p, id_to_label),
    )

    logger.info("Starting training")
    trainer.train()
    logger.info("Training complete. Saving model.")
    trainer.save_model(args.output_dir)
    # Save metadata about the training run (model id, labels, seed)
    try:
        metadata = {
            "model_name": args.model_name,
            "num_labels": len(labels),
            "labels": labels,
            "seed": args.seed,
        }
        meta_path = os.path.join(args.output_dir, "train_metadata.json")
        with open(meta_path, "w", encoding="utf-8") as mf:
            json.dump(metadata, mf, indent=2, ensure_ascii=False)
        logger.info(f"Saved training metadata to {meta_path}")
    except Exception as e:
        logger.warning(f"Could not save training metadata: {e}")

    # Evaluate on test file and save a detailed classification report
    try:
        logger.info(f"Reading test file: {args.test_file}")
        test_examples = read_conll(args.test_file)
        logger.info("Tokenizing and aligning labels (test)")
        test_tokenized = tokenize_and_align_labels(test_examples, tokenizer, label_to_id, max_length=args.max_length)
        test_dataset = Dataset.from_dict(test_tokenized)

        logger.info("Running prediction on test set")
        predictions, label_ids, metrics = trainer.predict(test_dataset)
        preds = np.argmax(predictions, axis=2)

        # convert preds/labels to tag sequences
        true_predictions = []
        true_labels = []
        for pred_row, label_row in zip(preds, label_ids):
            pred_tags = []
            true_tags = []
            for p_id, l_id in zip(pred_row, label_row):
                if l_id == -100:
                    continue
                pred_tags.append(id_to_label[int(p_id)])
                true_tags.append(id_to_label[int(l_id)])
            true_predictions.append(pred_tags)
            true_labels.append(true_tags)

        report = classification_report(true_labels, true_predictions)
        report_path = os.path.join(args.output_dir, "test_classification_report.txt")
        os.makedirs(args.output_dir, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as outf:
            outf.write(report)
        logger.info(f"Saved test classification report to {report_path}")

        # Save basic metrics returned by predict
        metrics_path = os.path.join(args.output_dir, "test_metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as mf:
            json.dump(metrics, mf, indent=2)
        logger.info(f"Saved test metrics to {metrics_path}")
    except Exception as e:
        logger.warning(f"Could not run test evaluation: {e}")


if __name__ == "__main__":
    main()
