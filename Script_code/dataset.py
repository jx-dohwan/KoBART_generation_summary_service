import argparse
import os
import glob
import torch
import ast
import numpy as np
import pandas as pd
from tqdm import tqdm, trange
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from functools import partial
from typing import Dict, List, Optional, Tuple
import json
import re
from transformers.tokenization_utils import PreTrainedTokenizerBase


class KoBARTSummaryDataset(Dataset):
    def __init__(self, file, tokenizer, text_max_len, summary_max_len,  ignore_index=-100):
        super().__init__()
        self.tokenizer = tokenizer
        self.text_max_len = text_max_len
        self.summary_max_len = summary_max_len
        self.docs = file
        self.docs = pd.read_csv(file, sep='\t')
        self.len = self.docs.shape[0]

        self.pad_index = self.tokenizer.pad_token_id
        self.ignore_index = ignore_index


    def add_ignored_data(self, inputs):
        if len(inputs) < self.summary_max_len:
            pad = [self.ignore_index] * (self.summary_max_len - len(inputs)) # ignore_index즉 -100으로 패딩을 만들 것인데 max_len - lne(inpu)
            inputs = np.concatenate([inputs, pad])
        else:
            inputs = inputs[:self.summary_max_len]

        return inputs

    def add_padding_data(self, inputs, is_summary=False):
        if is_summary == False:
            max_len = self.text_max_len
        else:
            max_len = self.summary_max_len

        pad_index = self.pad_index

        if len(inputs) < max_len:
            pad = [pad_index] *(max_len - len(inputs))
            inputs = np.concatenate([inputs, pad])
        else:
            inputs = inputs[:max_len]

        return inputs 

    def __getitem__(self, idx):
        print("idx",idx)
        instance = self.docs.iloc[idx]
        input_ids = self.tokenizer.encode(instance['dialogues'], add_special_tokens=False)
        input_ids = self.add_padding_data(input_ids)

        label_ids = self.tokenizer.encode(instance['summaries'])
        label_ids.append(self.tokenizer.eos_token_id)
        dec_input_ids = self.tokenizer('')['input_ids']
        dec_input_ids += label_ids[:-1]
        dec_input_ids = self.add_padding_data(dec_input_ids, is_summary=True)
        label_ids = self.add_ignored_data(label_ids)
        print("input_ids :", input_ids )
        print("label_ids", label_ids)
        print("dec_input_ids", dec_input_ids)
        return {'input_ids': np.array(input_ids, dtype=np.int_),
                'decoder_input_ids': np.array(dec_input_ids, dtype=np.int_),
                'labels': np.array(label_ids, dtype=np.int_)}

    def __len__(self):
        return self.len



class KoBARTSummaryModule(pl.LightningDataModule):
    def __init__(self,
                train_file,
                test_file,
                tok,
                text_max_len=256,
                summary_max_len=64,
                batch_size=256,
                num_workers=4):
        super().__init__()
        self.batch_size = batch_size
        self.text_max_len = text_max_len
        self.summary_max_len = summary_max_len
        self.train_file_path = train_file
        self.test_file_path = test_file
        self.tok = tok
        self.num_workers = num_workers

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(
            parents = [parent_parser], add_help=False            
        )
        parser.add_argument('--num_workers',
                            type=int,
                            default=4,
                            help='num of worker for dataloader')
        return parser

    # OPTIONAL, called for every GPU/machine(assingning state is OK)
    def setup(self, stage):
        # split dataset
        self.train = KoBARTSummaryDataset(
                                        self.train_file_path,
                                        self.tok,
                                        self.text_max_len,
                                        self.summary_max_len

                    )
        self.test = KoBARTSummaryDataset(
                                        self.test_file_path,
                                        self.tok,
                                        self.text_max_len,
                                        self.summary_max_len
                    )

    def train_dataloader(self):
        train = DataLoader(
                        self.train, 
                        batch_size=self.batch_size,
                        num_workers=self.num_workers, shuffle=True        
                )
        return train

    
    def val_dataloader(self):
        val = DataLoader(
                        self.test, 
                        batch_size=self.batch_size,
                        num_workers=self.num_workers, shuffle=False        
                )
        return val

    def test_dataloader(self):
        test = DataLoader(
                        self.test, 
                        batch_size=self.batch_size,
                        num_workers=self.num_workers, shuffle=False        
                )
        return test
        