import argparse
import json
import logging
import math
import os
import random
from pathlib import Path

import transformers
from transformers import (
    SchedulerType,
)

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
    parser.add_argument(
        "--task_name",
        type=str,
        default="sst2",
        help="The name of the glue task to train on.",
        choices=list(task_to_keys.keys()),
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=64,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_length` is passed."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    
    # teacher Model and student Model
    
    parser.add_argument(
        "--generator_model_name",
        type=str,
        default="roberta-base",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        # required=True,
    )
    parser.add_argument(
        "--teacher_model_name",
        type=str,
        default="roberta-base",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    
    parser.add_argument(
        "--student_model_name",
        type=str,
        default="distilroberta-base",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    
    # experiment arguments
    parser.add_argument(
        "--mlm_probability",
        type=float,
        default=0.45,
        help="The MLM prob in the data collator for LM",
    )
    parser.add_argument(
        "--select_k_per_class",
        type=int,
        default=16,
        help="few-shot setting for k"
    )
    parser.add_argument(
        "--intermediate_hidden_size",
        type=int,
        default=128,
        help="Project intermiediate size into same dimension.",
    )
    
    # training arguments
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for the training dataloader.",
    )
    
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--generator_lr",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--teacher_lr",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--student_lr",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--teacher_num_train_epochs", type=int, default=5, help="Total number of training epochs to perform.")
    parser.add_argument("--generator_num_train_epochs", type=int, default=5, help="Total number of training epochs to perform.")
    parser.add_argument("--student_num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )

    parser.add_argument(
        "--eval_every_step",
        type=int,
        default=100,
        help="do evaluation every steps",
    )

    parser.add_argument(
        "--teacher_checkpoint_path",
        type=str,
        default=None,
        help="loading pretrained teacher checkpoint path",
    )
    parser.add_argument(
        "--generator_checkpoint_path",
        type=str,
        default=None,
        help="loading pretrained teacher checkpoint path",
    )
        
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--ignore_mismatched_sizes",
        action="store_true",
        help="Whether or not to enable to load a pretrained model whose head dimensions are different.",
    )
    # parser.add_argument('--dataset', type=str, default='sst2')
    args = parser.parse_args()
    
    return args