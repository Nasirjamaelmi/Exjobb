#!/usr/bin/env python3
"""Simple script to run a saved token-classification model on a prompt.

Usage examples:
  python use_model_prompt.py --model_dir ner-output --prompt "John lives in New York City."
  python use_model_prompt.py --model_dir ner-output --input_file prompts.txt --output_file results.json

The script uses the Transformers `pipeline('ner')` to produce entity spans.
"""
import argparse
import json
import logging
import os
from typing import List, Any, Dict

from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_prompt(model_dir: str, prompt: str, grouped: bool = True) -> List[Dict[str, Any]]:
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(model_dir)

    # pipeline API changed names across versions; try common options
    try:
        if grouped:
            nlp = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
        else:
            nlp = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy=None)
    except TypeError:
        # Fallback for older Transformers where `grouped_entities` is used
        try:
            nlp = pipeline("ner", model=model, tokenizer=tokenizer, grouped_entities=grouped)
        except Exception:
            # Final fallback: pipeline with defaults
            nlp = pipeline("ner", model=model, tokenizer=tokenizer)

    ents = nlp(prompt)
    return ents


def build_label_map(model_dir: str, model) -> dict:
    """Return a mapping from pipeline label names (e.g. 'LABEL_3') to human labels.

    It will try, in order:
    - to load `train_metadata.json` from `model_dir` and use its `labels` list;
    - fall back to `model.config.id2label` values (may be LABEL_* strings).
    """
    lm = {}
    meta_path = os.path.join(model_dir, "train_metadata.json")
    id2label = getattr(model.config, "id2label", {}) or {}
    # Try metadata file first
    if os.path.exists(meta_path):
        try:
            with open(meta_path, encoding="utf-8") as mf:
                meta = json.load(mf)
            labels = meta.get("labels")
            if isinstance(labels, list):
                # id2label keys may be strings in saved config
                for i, lab in enumerate(labels):
                    key = id2label.get(str(i), id2label.get(i, f"LABEL_{i}"))
                    lm[key] = lab
                return lm
        except Exception:
            pass

    # Fallback: return identity mapping from id2label
    for k, v in id2label.items():
        lm[str(v)] = v
    return lm


def process_lines(model_dir: str, lines: List[str], grouped: bool, label_map: dict | None = None) -> List[Dict[str, Any]]:
    all_results = []
    # Load model to build mapping if not provided
    model = None
    if label_map is None:
        try:
            from transformers import AutoModelForTokenClassification
            model = AutoModelForTokenClassification.from_pretrained(model_dir)
            label_map = build_label_map(model_dir, model)
        except Exception:
            label_map = {}

    for i, line in enumerate(lines):
        text = line.strip()
        if not text:
            continue
        logger.info(f"Tagging prompt #{i+1}: {text}")
        ents = run_prompt(model_dir, text, grouped=grouped)
        # Convert float32 scores to native Python floats for JSON serialization
        # Also convert label keys (entity_group/entity/label) using label_map if available
        new_ents = []
        for e in ents:
            ne = dict(e)
            if "score" in ne:
                ne["score"] = float(ne["score"])
            # normalize possible label keys returned by different pipeline versions
            label_key = None
            for k in ("entity_group", "entity", "label"):
                if k in ne:
                    label_key = k
                    break
            if label_key is not None:
                raw_label = ne[label_key]
                mapped = label_map.get(str(raw_label), label_map.get(raw_label))
                if mapped:
                    ne["label"] = mapped
                else:
                    ne["label"] = raw_label
            new_ents.append(ne)
        all_results.append({"text": text, "entities": new_ents})
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Run a saved NER model on a prompt or file of prompts")
    parser.add_argument("--model_dir", default="ner-output", help="Directory with saved model/tokenizer")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prompt", type=str, help="Single prompt/sentence to tag")
    group.add_argument("--input_file", type=str, help="File with one prompt per line")
    parser.add_argument("--output_file", type=str, default=None, help="Path to JSON file to save results")
    parser.add_argument("--no_group", action="store_true", help="Do not group tokens into entity spans (use token-level outputs)")
    parser.add_argument("--label_map", type=str, default=None, help="Optional JSON file mapping label ids or LABEL_x names to readable labels (or leave to auto-load train_metadata.json)")
    args = parser.parse_args()

    if not os.path.isdir(args.model_dir):
        logger.error(f"Model directory does not exist: {args.model_dir}")
        raise SystemExit(1)

    grouped = not args.no_group

    # Load label map if provided
    label_map = None
    if args.label_map:
        if not os.path.exists(args.label_map):
            logger.error(f"Label map file not found: {args.label_map}")
            raise SystemExit(1)
        try:
            with open(args.label_map, encoding="utf-8") as lf:
                lm = json.load(lf)
            # If list -> convert to dict keyed by LABEL_{i} where possible
            if isinstance(lm, list):
                label_map = {}
                for i, name in enumerate(lm):
                    key = f"LABEL_{i}"
                    label_map[key] = name
            elif isinstance(lm, dict):
                label_map = lm
        except Exception as e:
            logger.error(f"Could not load label map file: {e}")
            raise SystemExit(1)

    if args.prompt:
        results = process_lines(args.model_dir, [args.prompt], grouped=grouped, label_map=label_map)
    else:
        if not os.path.exists(args.input_file):
            logger.error(f"Input file not found: {args.input_file}")
            raise SystemExit(1)
        with open(args.input_file, encoding="utf-8") as f:
            lines = f.readlines()
        results = process_lines(args.model_dir, lines, grouped=grouped, label_map=label_map)

    # Print to stdout
    print(json.dumps(results, indent=2, ensure_ascii=False))

    # Optionally save
    if args.output_file:
        with open(args.output_file, "w", encoding="utf-8") as outf:
            json.dump(results, outf, indent=2, ensure_ascii=False)
        logger.info(f"Saved results to {args.output_file}")


if __name__ == "__main__":
    main()
