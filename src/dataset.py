import pandas as pd
import numpy as np

from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import datasets
from transformers import (
    DataCollatorWithPadding,
    DataCollatorForLanguageModeling,
)

from src.datacollator import DataCollatorForLanguageModelingAndClassification

class GLUE_Dataset:
    def __init__(self, args, tokenizer) -> None:
        self.args = args
        self.tokenizer = tokenizer
        self.max_length = args.max_length
        self.task_to_keys = {
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
        
        self.label_key = 'label'
        
        self.dataset = load_dataset('glue', args.task_name)
        print(self.dataset)
        
    def select_subset_ds(self, ds, k=200, seed=0):
    
        label_key = self.label_key
        N = len(ds[label_key])
        idx_total = np.array([], dtype='int64')

        for l in set(ds[label_key]):
            idx = np.where(np.array(ds[label_key]) == l)[0]
            idx_total = np.concatenate([idx_total, # we cannot take more samples than there are available
            np.random.choice(idx, min(k, idx.shape[0]), replace=False)])

        np.random.seed(seed)
        np.random.shuffle(idx_total)
        return ds.select(idx_total)
    
    def preprocess_fn(self, examples, task):
        tokenizer = self.tokenizer
        # Preprocess dataset
        sentence1_key, sentence2_key = self.task_to_keys[task]
        padding = "max_length"
        text = (
            (examples[sentence1_key]) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        
        inputs = tokenizer(text, padding='max_length', max_length=self.max_length, truncation=True)
        
        inputs['labels'] = examples['label']
        return inputs

    def get_final_ds(self, task, split, batch_size, k=-1, seed=0):
        # model_type : ['gen' , 'clf']
        dataset = self.dataset[split]
        if k!=-1:
            dataset = self.select_subset_ds(dataset, k)
        else:
            dataset = dataset.shuffle(seed=seed)
        
        processed_dataset = dataset.map(
            lambda x: self.preprocess_fn(x, task),
            batched=True,
        )
        
        data_collator = DataCollatorForLanguageModelingAndClassification(
            tokenizer=self.tokenizer, mlm_probability=self.args.mlm_probability, pad_to_multiple_of=8
        )

        processed_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
        
        dataloader = DataLoader(processed_dataset, collate_fn=data_collator, batch_size=batch_size)
        return dataloader