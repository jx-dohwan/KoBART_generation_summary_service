import random
import datasets
import transformers
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import itertools
import re
import os
import json
from datasets import Dataset
from rouge import Rouge
from transformers import (
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    AutoModelForMaskedLM


)
from functools import partial
from tqdm import tqdm
import torch
import argparse

import sys



class post_train:
    def __init__(self, config):
        self.tokenizer = AutoTokenizer.from_pretrained(config.checkpoint)
        self.model = AutoModelForMaskedLM.from_pretrained(config.checkpoint)

        self.special_words = [
                        "#@주소#", "#@이모티콘#", "#@이름#", "#@URL#", "#@소속#",
                        "#@기타#", "#@전번#", "#@계정#", "#@url#", "#@번호#", "#@금융#", "#@신원#",
                        "#@장소#", "#@시스템#사진#", "#@시스템#동영상#", "#@시스템#기타#", "#@시스템#검색#",
                        "#@시스템#지도#", "#@시스템#삭제#", "#@시스템#파일#", "#@시스템#송금#", "#@시스템#",
                        "#개인 및 관계#", "#미용과 건강#", "#상거래(쇼핑)#", "#시사/교육#", "#식음료#", 
                        "#여가 생활#", "#일과 직업#", "#주거와 생활#", "#행사#","[sep]"
                    ]
    
        self.tokenizer.add_special_tokens({"additional_special_tokens": self.special_words})
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.config.max_length = 128 
        self.model.config.early_stopping = True
        self.model.config.no_repeat_ngram_size = 3
        self.model.config.length_penalty = 2.0
        self.model.config.num_beams = 5

        self.config = config
    """ 데이터 불러오기 및 전처리리"""
    def data_mining(data):
        body_list = []
        text_list = []
        total_list = []
        with open(data) as f:
              data = json.load(f)
        for datum in data["data"]:
            body_list.append(datum['body'])  
        for text in body_list:
          text_list.append(text)
        for utt in text_list:
            utt_list = []
            for i in range(len(utt)):
                utt_list.append(utt[i]['utterance'])
            total_list.append(utt_list)
        return total_list

    def data_load(filename, data_mining):
        data_mining = data_mining()
        text = []

        for file in tqdm(filename):
          total_list = data_mining(file)
          for data in total_list:
            text.append("[sep]".join(data))

        return text

    def preprocess_sentence(sentence):
        sentence = sentence.lower() # 텍스트 소문자화
        sentence = re.sub(r'[ㄱ-ㅎㅏ-ㅣ]+[/ㄱ-ㅎㅏ-ㅣ]', '', sentence) # 여러개 자음과 모음을 삭제한다.
        sentence = re.sub("[^가-힣a-z0-9#@,-\[\]\(\)]", " ", sentence) # 영어 외 문자(숫자, 특수문자 등) 공백으로 변환
        sentence = re.sub(r'[" "]+', " ", sentence) # 여러개 공백을 하나의 공백으로 바꿉니다.
        sentence = sentence.strip() # 문장 양쪽 공백 제거

        return sentence

    def data_process(data, preprocess_sentence):
        preprocess_sentence = preprocess_sentence()
        # 전체 Text 데이터에 대한 전처리 (1)
        text = []

        for data_text in tqdm(data):
          text.append(preprocess_sentence(data_text))
        
        return text

    """ Tokenizer """
    def add_ignored_data(self, inputs):
      if len(inputs) < config.max_len:
          pad = [self.config.ignore_index] *(self.config.max_len - len(inputs)) # ignore_index즉 -100으로 패딩을 만들 것인데 max_len - lne(inpu)
          inputs = np.concatenate([inputs, pad])
      else:
          inputs = inputs[:self.config.max_len]

      return inputs

    def add_padding_data(self, inputs, is_masking=False):
        if is_masking:
            mask_num = int(len(inputs)*self.config.masking_rate)
            mask_positions = random.sample([x for x in range(len(inputs))], mask_num)
            corrupt_token = []
            for pos in range(len(inputs)):
                if pos in mask_positions:
                    corrupt_token.append(self.tokenizer.mask_token_id)
                else:
                    corrupt_token(inputs[pos])

        if len(corrupt_token) < self.config.max_len:
            pad = [self.tokenizer.pad_token_id] * (self.config.max_len - len(corrupt_token))
            inputs = np.concatenate([corrupt_token, pad])
        else:
            inputs = corrupt_token[:self.config.max_len]

        return inputs 


    def preprocess_data(self, data_to_process, add_ignored_data, add_padding_data):
        add_ignored_data = add_ignored_data()
        add_padding_data = add_padding_data()

        label_id= []
        label_ids = []
        dec_input_ids = []
        input_ids = []
        bos = self.tokenizer('<s>')['input_ids']
        for i in range(len(data_to_process['Text'])):
            input_ids.append(add_padding_data(self.tokenizer.encode(data_to_process['Text'][i], add_special_tokens=False),  is_masking=True))
        for i in range(len(data_to_process['Text'])):
            label_id.append(self.tokenizer.encode(data_to_process['Text'][i]))  
            label_id[i].append(self.tokenizer.eos_token_id)   
            dec_input_id = bos
            dec_input_id += label_id[i][:-1]
            dec_input_ids.append(add_padding_data(dec_input_id))  
        for i in range(len(data_to_process['Text'])):
            label_ids.append(add_ignored_data(label_id[i]))

        return {'input_ids': input_ids,
                'attention_mask' : (np.array(input_ids) != self.tokenizer.pad_token_id).astype(int),
                'decoder_input_ids': dec_input_ids,
                'decoder_attention_mask': (np.array(dec_input_ids) != self.tokenizer.pad_token_id).astype(int),
                'labels': label_ids}

    def main(self,data_load, data_process, preprocess_data):
        data_load = data_load()
        data_process = data_process()
        preprocess_data = preprocess_data()

        """데이터 불러오기기 및 전처리 및 Dataset으로로 변환""" 
        filenames_train = os.listdir(self.config.train_fn) 
        train_full_filename = []
        for filename in filenames_train:
            tfn = os.path.join(self.config.train_fn, filename)
            if self.config.train_fn.train_fnme + '/.ipynb_checkpoints' != tfn:
                train_full_filename.append(tfn)

        filenames_valid = os.listdir(self.config.valid_fn) 
        val_full_filename = []
        for filename in filenames_valid:
            vfn = os.path.join(self.config.valid_fn, filename)
            if self.config.valid_fn + '/.ipynb_checkpoints' != vfn:
                val_full_filename.append(vfn)

        train_dataset = data_load(train_full_filename)
        train_list = data_process(train_dataset)

        val_dataset = data_load(val_full_filename)
        val_list = data_process(val_dataset)

        train_df = pd.DataFrame(zip(train_list), columns=['Text'])
        val_df = pd.DataFrame(zip(val_list), columns=['Text'])

        train_data = Dataset.from_pandas(train_df) 
        val_data = Dataset.from_pandas(val_df)
        

        """Tokenizer"""
        preprocess_data = partial(preprocess_data, add_ignored_data=True, add_padding_data=True)
        
        train_tokenize_data = train_data.map(preprocess_data, batched = True, remove_columns=['Text'])
        val_tokenize_data = val_data.map(preprocess_data, batched = True, remove_columns=['Text'])
    
    


def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument('--checkpoint', required=True)
    p.add_argument('--train_fn', required=True)
    p.add_argument('--valid_fn', required=True)
    p.add_argument('--save_fn', required=True)

    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--ignore_index', type=int, default=-100)
    p.add_argument('--max_len', type=int, default=128)
    p.add_argument('--train_batch_size', type=int, default=128)
    p.add_argument('--valid_batch_size', type=int, default=256)
    p.add_argument('--lr', type=float, default=3e-05)
    p.add_argument('--weight_decay', type=float, default=0.1)
    p.add_argument('--masking_rate', type=float, default=0.15)
    p.add_argument('--save_limit', type=int, default=3)
    p.add_argument('--load_best_model', type=bool, default=True)


if __name__ == '__main__':
    config = define_argparser()
    post_train(config)