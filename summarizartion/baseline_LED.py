#!/usr/bin/env python
# coding=utf-8
"""
Fine-tune the Longformer Encoder-Decoder (LED) model on MIMIC-IV clinical note summarization.
"""

import os
import sys
import logging
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

import torch
import nltk
import wandb
from datasets import load_dataset
from rouge_score import rouge_scorer
import evaluate
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Download NLTK resources
try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    nltk.download("punkt", quiet=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune LED model for MIMIC-IV summarization")
    # Model arguments
    parser.add_argument("--model_name_or_path", type=str, default="allenai/led-base-16384",
                        help="Path to pretrained model or model identifier from huggingface.co/models")
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="Where to store the pretrained models downloaded from huggingface.co")
    parser.add_argument("--use_fast_tokenizer", action="store_true",
                        help="Whether to use fast tokenizer")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout probability for the model")
    
    # Data arguments
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to the dataset directory")
    parser.add_argument("--max_source_length", type=int, default=4096,
                        help="Maximum source sequence length")
    parser.add_argument("--max_target_length", type=int, default=600,
                        help="Maximum target sequence length")
    parser.add_argument("--num_train_examples", type=int, default=None,
                        help="Limit the number of training examples")
    parser.add_argument("--num_val_examples", type=int, default=None,
                        help="Limit the number of validation examples")
    parser.add_argument("--num_test_examples", type=int, default=None,
                        help="Limit the number of test examples")
    parser.add_argument("--preprocessing_num_workers", type=int, default=4,
                        help="Number of processes for preprocessing")
    parser.add_argument("--overwrite_cache", action="store_true",
                        help="Overwrite the cached training and evaluation sets")
    
    # Training arguments
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save the model and results")
    parser.add_argument("--do_train", action="store_true",
                        help="Whether to run training")
    parser.add_argument("--do_eval", action="store_true",
                        help="Whether to run evaluation on the validation set")
    parser.add_argument("--do_predict", action="store_true",
                        help="Whether to run predictions on the test set")
    parser.add_argument("--per_device_train_batch_size", type=int, default=2,
                        help="Batch size per GPU/TPU core/CPU for training")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=2,
                        help="Batch size per GPU/TPU core/CPU for evaluation")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8,
                        help="Number of updates steps to accumulate before backward/update pass")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="The initial learning rate for AdamW")
    parser.add_argument("--weight_decay", type=float, default=0.0,
                        help="Weight decay for AdamW optimizer")
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Max number of training steps")
    parser.add_argument("--num_train_epochs", type=int, default=3,
                        help="Total number of training epochs to perform")
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                        help="Linear warmup over warmup_ratio fraction of total steps") 
    parser.add_argument("--logging_steps", type=int, default=100,
                        help="Log every X updates steps")
    parser.add_argument("--save_steps", type=int, default=500,
                        help="Save checkpoint every X updates steps")
    parser.add_argument("--eval_steps", type=int, default=500,
                        help="Run evaluation every X steps")
    parser.add_argument("--save_total_limit", type=int, default=3,
                        help="Limit the total amount of checkpoints")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for initialization")
    parser.add_argument("--num_beams", type=int, default=4,
                        help="Number of beams for beam search")
    
    # WandB arguments
    parser.add_argument("--wandb_project", type=str, default="mimic-iv-led-baseline",
                        help="WandB project name")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="WandB run name")
    
    args = parser.parse_args()
    return args


def preprocess_function(examples, tokenizer, args):
    """Preprocess the data by tokenizing."""
    inputs = examples["text"]
    targets = examples["summary"]
    
    model_inputs = tokenizer(
        inputs, 
        max_length=args.max_source_length, 
        padding="max_length" if args.pad_to_max_length else False, 
        truncation=True
    )
    
    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets,
            max_length=args.max_target_length,
            padding="max_length" if args.pad_to_max_length else False,
            truncation=True
        )
    
    # If padding, replace all tokenizer.pad_token_id in the labels by -100
    if args.pad_to_max_length:
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]
    
    model_inputs["labels"] = labels["input_ids"]
    
    # Fix for LED-large-16384
    if 'led-large-16384' in args.model_name_or_path:
        model_inputs["labels"] = [x[1:] for x in model_inputs["labels"]]
    
    return model_inputs


# Custom evaluation metrics
def get_rouge_score(gold, pred):
    rouge_scores = ['rouge1', 'rouge2', 'rouge3', 'rouge4', 'rougeL']
    scorer = rouge_scorer.RougeScorer(rouge_scores, use_stemmer=True)
    scores = scorer.score(gold, pred)
    return {k: scores[k].fmeasure * 100 for k in rouge_scores}


def compute_custom_metrics(srcs, golds, preds, device="cuda"):
    scores = defaultdict(list)
    bertscore = evaluate.load("bertscore")
    sari = evaluate.load("sari")
    
    # For rouge and length go over examples one by one and determine mean
    for gold, pred in zip(golds, preds):
        for k, v in get_rouge_score(gold, pred).items():
            scores[k].append(v)
        scores['words'].append(len(pred.split(' ')))
    
    for k, v in scores.items():
        scores[k] = np.mean(v)

    # BERTScore with default model (roberta-large)
    scores['bert_score'] = np.mean((bertscore.compute(
        predictions=preds, references=golds, lang="en", device=device))['f1']) * 100
    
    # BERTScore with recommended model (deberta-large-mnli)
    scores['bert_score_deberta-large'] = np.mean((bertscore.compute(
        predictions=preds, references=golds, device=device, model_type="microsoft/deberta-large-mnli"))['f1']) * 100
    
    # SARI score
    scores['sari'] = sari.compute(
        sources=srcs, predictions=preds, references=[[g] for g in golds])['sari']
    
    return scores


def print_metrics_as_latex(metrics):
    order = ['rouge1', 'rouge2', 'rouge3', 'rouge4', 'rougeL', 'bert_score', 'bert_score_deberta-large', 'sari', 'words']
    print(' & '.join([f'${metrics[k]:.2f}$' for k in order]))


def postprocess_text(preds, labels):
    """Clean up the generated text and reference for ROUGE evaluation."""
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]
    
    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]
    
    return preds, labels


def compute_metrics(eval_preds, tokenizer):
    """Compute ROUGE metrics for the trainer."""
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    
    # Replace -100 in the labels as we can't decode them
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
    
    # Load metric
    metric = evaluate.load("rouge")
    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    
    # Extract a few results
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    
    # Add mean generated length
    prediction_lens = [len(pred.split()) for pred in decoded_preds]
    result["gen_len"] = np.mean(prediction_lens)
    
    return {k: round(v, 4) for k, v in result.items()}


def main():
    args = parse_args()
    
    # Set up wandb
    if args.do_train:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or f"led-baseline-{args.model_name_or_path.split('/')[-1]}"
        )
    
    # Set seed
    set_seed(args.seed)
    
    # Load datasets
    data_files = {
        "train": os.path.join(args.data_path, "train_4000_600_chars_251-350_pt.json"),
        "validation": os.path.join(args.data_path, "valid_4000_600_chars.json"),
        "test": os.path.join(args.data_path, "valid_4000_600_chars.json")
    }
    datasets = load_dataset("json", data_files=data_files)
    
    # Limit dataset size if specified
    if args.num_train_examples and "train" in datasets:
        datasets["train"] = datasets["train"].select(range(-args.num_train_examples, 0))
    if args.num_val_examples and "validation" in datasets:
        datasets["validation"] = datasets["validation"].select(range(-args.num_val_examples, 0))
    if args.num_test_examples and "test" in datasets:
        datasets["test"] = datasets["test"].select(range(-args.num_test_examples, 0))
    
    # Print dataset sizes
    logger.info(f"Training set size: {len(datasets['train']) if 'train' in datasets else 0}")
    logger.info(f"Validation set size: {len(datasets['validation']) if 'validation' in datasets else 0}")
    logger.info(f"Test set size: {len(datasets['test']) if 'test' in datasets else 0}")
    
    # Load the model
    config = AutoConfig.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        dropout=args.dropout,
        trust_remote_code=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        use_fast=args.use_fast_tokenizer,
        trust_remote_code=True
    )
    
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_name_or_path,
        config=config,
        cache_dir=args.cache_dir,
        trust_remote_code=True
    )
    
    # Set padding parameters
    args.pad_to_max_length = False  # Dynamic padding is more efficient
    
    # Preprocessing datasets
    if args.do_train:
        column_names = datasets["train"].column_names
    elif args.do_eval:
        column_names = datasets["validation"].column_names
    elif args.do_predict:
        column_names = datasets["test"].column_names
    else:
        column_names = datasets[list(datasets.keys())[0]].column_names
    
    # Function for preprocessing
    def preprocess_dataset(examples):
        return preprocess_function(examples, tokenizer, args)
    
    # Process datasets
    if args.do_train:
        train_dataset = datasets["train"]
        train_dataset = train_dataset.map(
            preprocess_dataset,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on training dataset",
        )
    else:
        train_dataset = None
    
    if args.do_eval:
        eval_dataset = datasets["validation"]
        eval_dataset = eval_dataset.map(
            preprocess_dataset,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on validation dataset",
        )
    else:
        eval_dataset = None
    
    if args.do_predict:
        test_dataset = datasets["test"]
        orig_test_dataset = test_dataset
        test_dataset = test_dataset.map(
            preprocess_dataset,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on test dataset",
        )
    else:
        test_dataset = None
        orig_test_dataset = None
    
    # Data collator
    label_pad_token_id = -100
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if args.fp16 else None,
    )
    
    # Define training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        do_train=args.do_train,
        do_eval=args.do_eval,
        do_predict=args.do_predict,
        evaluation_strategy="steps" if args.do_eval else "no",
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_steps=args.max_steps,
        num_train_epochs=args.num_train_epochs if not args.max_steps else None,
        warmup_ratio=args.warmup_ratio,
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        eval_steps=args.eval_steps if args.do_eval else None,
        predict_with_generate=True,
        generation_max_length=args.max_target_length,
        generation_num_beams=args.num_beams,
        seed=args.seed,
        fp16=torch.cuda.is_available(),
        report_to=["wandb"] if wandb.run is not None else [],
        load_best_model_at_end=True if args.do_eval else False,
        metric_for_best_model="eval_rouge2" if args.do_eval else None,
        greater_is_better=True,
    )
    
    # Initialize the trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda eval_preds: compute_metrics(eval_preds, tokenizer),
    )
    
    # Training
    if args.do_train:
        logger.info("*** Training ***")
        train_result = trainer.train()
        trainer.save_model()
        
        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
    
    # Evaluation
    if args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(
            max_length=args.max_target_length,
            num_beams=args.num_beams,
            metric_key_prefix="eval"
        )
        metrics["eval_samples"] = len(eval_dataset)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
    
    # Prediction
    if args.do_predict:
        logger.info("*** Predict ***")
        predict_results = trainer.predict(
            test_dataset=test_dataset,
            max_length=args.max_target_length,
            num_beams=args.num_beams,
            metric_key_prefix="test"
        )
        metrics = predict_results.metrics
        metrics["test_samples"] = len(test_dataset)
        
        trainer.log_metrics("test", metrics)
        trainer.save_metrics("test", metrics)
        
        if predict_results.predictions is not None:
            test_preds = tokenizer.batch_decode(
                predict_results.predictions, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=True
            )
            test_preds = [pred.strip() for pred in test_preds]
            
            # Compute and log custom metrics
            metrics_test = compute_custom_metrics(
                srcs=orig_test_dataset["text"],
                golds=orig_test_dataset["summary"],
                preds=test_preds
            )
            print("Test metrics:")
            print(metrics_test)
            print_metrics_as_latex(metrics_test)
            wandb.log(metrics_test)
            
            # Save predictions
            output_file = os.path.join(args.output_dir, "test_generations.txt")
            with open(output_file, "w") as f:
                f.write("\n".join(test_preds))
                
            # Save JSON format
            import json
            output_json = os.path.join(args.output_dir, "test_generations.json")
            with open(output_json, "w") as f:
                json.dump([{"summary": pred} for pred in test_preds], f)
    
    wandb.finish()
    
    return metrics_test if args.do_predict else None


if __name__ == "__main__":
    main()