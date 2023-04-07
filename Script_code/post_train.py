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
    AutoModelForMaskedLM,



)
import torch.nn as nn
from functools import partial
from tqdm import tqdm
import torch
import argparse
import wandb


import sys
sys.path.append('/content/drive/MyDrive/인공지능/생성요약프로젝트/Model/script')
from post_dataset import data_load, data_process, preprocess_data               


def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument('--checkpoint', required=True)
    p.add_argument('--train_fn', required=True)
    p.add_argument('--valid_fn', required=True)
    p.add_argument('--save_fn', required=True)

    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--no_repeat_ngram_size', type=int, default=3)
    p.add_argument('--num_beams', type=int, default=5)
    p.add_argument('--length_penalty', type=float, default=2.0)
    p.add_argument('--ignore_index', type=int, default=-100)
    p.add_argument('--max_len', type=int, default=128)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--lr', type=float, default=3e-05)
    p.add_argument('--weight_decay', type=float, default=0.1)
    p.add_argument('--masking_rate', type=float, default=0.15)
    p.add_argument('--save_limit', type=int, default=3)
    p.add_argument('--load_best_model', type=bool, default=True)
    p.add_argument('--predict_with_generate', type=bool, default=True)
    p.add_argument('--do_train', type=bool, default=True)
    p.add_argument('--do_eval', type=bool, default=True)
    p.add_argument('--warmup_ratio', type=float, default=.1)

    config = p.parse_args()

    return config

def main(config):
  

    """data loader"""
    filenames_t = os.listdir(config.train_fn) 
    train_full_filename = []

    for filename in filenames_t:
        fnt = os.path.join(config.train_fn, filename)
        if config.train_fn + '/.ipynb_checkpoints' != fnt:
            train_full_filename.append(fnt)

    filenames_v = os.listdir(config.valid_fn) 
    val_full_filename = []

    for filename in filenames_v:
        fnv = os.path.join(config.valid_fn, filename)
        if config.valid_fn + '/.ipynb_checkpoints' != fnv:
            val_full_filename.append(fnv)

    train_dataset = data_load(train_full_filename, True)
    train_list = data_process(train_dataset)

    val_dataset = data_load(val_full_filename)
    val_list = data_process(val_dataset)

    train_df = pd.DataFrame(zip(train_list), columns=['Text'])
    val_df = pd.DataFrame(zip(val_list), columns=['Text'])

    train_data = Dataset.from_pandas(train_df) 
    val_data = Dataset.from_pandas(val_df)

    """tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained(config.checkpoint)
    model = AutoModelForMaskedLM.from_pretrained(config.checkpoint)

    special_words = [
                    "#@주소#", "#@이모티콘#", "#@이름#", "#@URL#", "#@소속#",
                    "#@기타#", "#@전번#", "#@계정#", "#@url#", "#@번호#", "#@금융#", "#@신원#",
                    "#@장소#", "#@시스템#사진#", "#@시스템#동영상#", "#@시스템#기타#", "#@시스템#검색#",
                    "#@시스템#지도#", "#@시스템#삭제#", "#@시스템#파일#", "#@시스템#송금#", "#@시스템#",
                    "#개인 및 관계#", "#미용과 건강#", "#상거래(쇼핑)#", "#시사/교육#", "#식음료#", 
                    "#여가 생활#", "#일과 직업#", "#주거와 생활#", "#행사#","[sep]"
                    ]

    tokenizer.add_special_tokens({"additional_special_tokens": special_words})

    train_tokenize_data = train_data.map(partial(preprocess_data, config=config, tokenizer=tokenizer), batched = True, remove_columns=['Text'])
    val_tokenize_data = val_data.map(partial(preprocess_data, config=config, tokenizer=tokenizer), batched = True, remove_columns=['Text'])
    print(train_tokenize_data[0])
    print(val_tokenize_data[0])
    """model"""
    model.resize_token_embeddings(len(tokenizer))
    model.config.max_length = config.max_len 
    model.config.no_repeat_ngram_size = config.no_repeat_ngram_size
    model.config.length_penalty = config.length_penalty
    model.config.num_beams = config.num_beams

    """ train """
    n_warmup_steps = int(len(train_tokenize_data) * config.warmup_ratio)
   
    training_args = Seq2SeqTrainingArguments(
        output_dir=config.save_fn + "/checkpoint",
        num_train_epochs=config.epochs,  # demo
        do_train=config.do_train,
        do_eval=config.do_eval,
        per_device_train_batch_size=config.batch_size,  
        per_device_eval_batch_size=config.batch_size,
        learning_rate=config.lr,
        weight_decay=config.weight_decay,
        predict_with_generate=config.predict_with_generate, # 생성기능을 사용하고 싶다고 지정한다.
        logging_dir=config.save_fn + "/log",
        save_total_limit=config.save_limit,
        load_best_model_at_end = config.load_best_model,
        logging_strategy = 'epoch',
        evaluation_strategy  = 'epoch',
        save_strategy ='epoch',
        warmup_steps=n_warmup_steps,

    )
    

    trainer = Seq2SeqTrainer(
        model, 
        training_args,
        train_dataset=train_tokenize_data,
        eval_dataset=val_tokenize_data,
        tokenizer=tokenizer,
     
    )
    
    trainer.train()
  



if __name__ == '__main__':
    config = define_argparser()
    main(config)