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

def load_json_data(path):

    with open(path) as f:
        data = json.load(f)

    ids = []
    dialogues = []
    summaries = []
    topic = []
    for datum in data["data"]:
        ids.append(datum["header"]["dialogueInfo"]["dialogueID"])

        prev_speaker_id = None
        prev_line = ""
        utts = []
        for dialogue in datum["body"]["dialogue"]:
            utterance = dialogue["utterance"].strip()

            if dialogue["participantID"] == prev_speaker_id:
                prev_line += " " + utterance
            else:
                if prev_line:
                    utts.append(prev_line)
                prev_line = utterance
                prev_speaker_id = dialogue["participantID"]
        if prev_line:
            utts.append(prev_line)

        dialogues.append(utts)
        summaries.append(datum["body"].get("summary"))

    for i in range(len(data['data'])):
      topic.append(data['data'][i]['header']['dialogueInfo']['topic'])
    return ids, dialogues, summaries, topic

def data_load(filename, is_meta=False):
    ids_list, dialogues_list, summaries_list, topic_list = [], [], [], []
    dialogues_sep = []

    for file in tqdm(filename):
      ids, dialogues, summaries, topic = load_json_data(file)
      for id, text, summ, top in zip(ids, dialogues, summaries, topic):
        ids_list.append(id)
        if is_meta:
          text.insert(0,"#"+top+"#")
        dialogues_list.append(text)
        summaries_list.append(summ)
        topic_list.append(top)
    
    for text in tqdm(dialogues_list):
      dialogues_sep.append("[sep]".join(text))

    return ids_list, dialogues_sep, summaries_list


class KoBARTSummaryDataset(Dataset):
    def __init__(self, file, tokenizer, max_len, ignore_index=-100):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.docs = pd.read_csv(file, sep='/t')
        self.len = self.docs.shape[0]

        self.pad_index = self.tokenizer.pad_token_id
        self.ignore__index = ignore_index

    def add_padding_data(self, inputs):
        if len(inputs) < self.max_len:
            pad = np.array([self.pad_index] * (self.max_len - len(inputs)))
            inputs = np.concatenate([inputs, pad])
        else:
            inputs = inputs[:self.max_len]

        return inputs

    def add_ignored_data(self, inputs):
        if len(inputs) < self.max_len:
            pad = np.array([self.ignore_index] * (self.max_len - len(inputs)))