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

def preprocess_sentence(sentence):
    sentence = sentence.lower() # 텍스트 소문자화
    sentence = re.sub(r'[ㄱ-ㅎㅏ-ㅣ]+[/ㄱ-ㅎㅏ-ㅣ]', '', sentence) # 여러개 자음과 모음을 삭제한다.
    sentence = re.sub("[^가-힣a-z0-9#@,-]", " ", sentence) # 영어 외 문자(숫자, 특수문자 등) 공백으로 변환
    sentence = re.sub(r'[" "]+', " ", sentence) # 여러개 공백을 하나의 공백으로 바꿉니다.
    sentence = sentence.strip() # 문장 양쪽 공백 제거
    
    return sentence

def path(file):
    filenames = os.listdir(file) 
    full_filename = []

    for filename in filenames:
        fn2 = os.path.join(file, filename)
        full_filename.append(fn2)

    return full_filename

class KoBARTSummaryDataset(Dataset):
    def __init__(self, file, tokenizer, max_len, ignore_index=-100):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.docs = file
        self.docs = pd.read_csv(file, sep='\t')
        self.len = self.docs.shape[0]

        self.pad_index = self.tokenizer.pad_token_id
        self.ignore_index = ignore_index

    def add_ignored_data(inputs, max_len, ignore_index):
        if len(inputs) < max_len:
            pad = [ignore_index] *(max_len - len(inputs)) # ignore_index즉 -100으로 패딩을 만들 것인데 max_len - lne(inpu)
            inputs = np.concatenate([inputs, pad])
        else:
            inputs = inputs[:max_len]

        return inputs

    def add_padding_data(inputs, max_len):
        pad_index = pad_index
        if len(inputs) < max_len:
            pad = [pad_index] *(max_len - len(inputs))
            inputs = np.concatenate([inputs, pad])
        else:
            inputs = inputs[:max_len]

        return inputs 

    def __getitem__(self, idx):

@staticmethod
class KoBARTSummaryModule(pl.LightningDataModule):
    def __init__(self,
                train_file,
                test_file,
                tok,
                max_len=256,
                batch_size=256)
        