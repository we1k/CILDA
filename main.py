
import argparse
import json
import logging
import math
import os
import random
from pathlib import Path

import datasets
import evaluate
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers

# from src.model import CILDA
from src.Args import parse_args
from src import CILDA

logger = get_logger(__name__)

def main():

    # setting for logger
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    datasets.utils.logging.set_verbosity_warning()
    transformers.utils.logging.set_verbosity_info()
    args = parse_args()
    cil_model = CILDA(args)
    # First do training on the teacher using full dataset
    
    if args.do_train_teacher:
        cil_model.train_teacher(train_epochs=args.teacher_num_train_epochs)
    
    # train the generator using few-shot dataset
    if args.do_train_generator:
        cil_model.train_generator(train_epochs=args.generator_num_train_epochs) 

    # generate synthetic data
    if args.generate_data:
        cil_model.generate_synthetic_data(syn_data_output_path='data/syn_data.json')
    
    # train student using generate_synthetic and few-shot data 
    if args.do_train_student:
        cil_model.train_student(train_epochs=args.student_num_train_epochs)
    
if __name__ == '__main__':
    main()