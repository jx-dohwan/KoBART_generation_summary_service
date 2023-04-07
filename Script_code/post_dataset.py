from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForMaskedLM
import json
from tqdm import tqdm
import re
import numpy as np 
import random
import torch

def data_mining(data):
  text_list = []
  total_list = []
  topic_list = []
  with open(data) as f:
        data = json.load(f)
  for datum in data["data"]:
      text_list.append(datum['body'])  
      topic_list.append(datum['header']['dialogueInfo']['topic'])
  for utt in text_list:
      utt_list = []
      for i in range(len(utt)):
          utt_list.append(utt[i]['utterance'])
      total_list.append(utt_list)

  return total_list, topic_list

def data_load(filename, is_meta=False):
    text = []
    for file in tqdm(filename):
      total_list, topic_list = data_mining(file)
      for i in range(len(total_list)):
        if is_meta:
          total_list[i].insert(0,"#"+topic_list[i]+"#")
        text.append("[sep]".join(total_list[i]))
    
    return text


def preprocess_sentence(sentence):
    sentence = sentence.lower() # 텍스트 소문자화
    sentence = re.sub(r'[ㄱ-ㅎㅏ-ㅣ]+[/ㄱ-ㅎㅏ-ㅣ]', '', sentence) # 여러개 자음과 모음을 삭제한다.
    sentence = re.sub("[^가-힣a-z0-9#@,-\[\]\(\)]", " ", sentence) # 영어 외 문자(숫자, 특수문자 등) 공백으로 변환
    sentence = re.sub(r'[" "]+', " ", sentence) # 여러개 공백을 하나의 공백으로 바꿉니다.
    sentence = sentence.strip() # 문장 양쪽 공백 제거
    
    return sentence

def data_process(data):
    # 전체 Text 데이터에 대한 전처리 (1)
    text = []

    for data_text in tqdm(data):
      text.append(preprocess_sentence(data_text))
    
    return text


def add_ignored_data(inputs, config, corrupt_token, tokenizer):
  none_mask = []
  corrupt_token = [x for x in corrupt_token if x != tokenizer.pad_token_id]
  for i in range(len(corrupt_token)):
      if corrupt_token[i] != tokenizer.mask_token_id:
          none_mask.append(i)
  for mask_num in none_mask:
      inputs[mask_num] = config.ignore_index
  if len(inputs) < config.max_len:
      pad = [config.ignore_index] *(config.max_len - len(inputs)) # ignore_index즉 -100으로 패딩을 만들 것인데 max_len - lne(inpu)
      inputs = np.concatenate([inputs, pad])
  else:
      inputs = inputs[:config.max_len]

  return inputs

def add_padding_data(inputs, config, tokenizer, is_mlm=False):

    if is_mlm:
        mask_num = int(len(inputs)*config.masking_rate)
        mask_positions = random.sample([x for x in range(len(inputs))], mask_num)

        corrupt_token = []

        for pos in range(len(inputs)):  
            if pos in mask_positions:           
                corrupt_token.append(tokenizer.mask_token_id)               
            else:
                corrupt_token.append(inputs[pos])

        if len(corrupt_token) < config.max_len:
            pad = [tokenizer.pad_token_id] * (config.max_len - len(corrupt_token))
            inputs = np.concatenate([corrupt_token, pad])
        else:
            inputs = corrupt_token[:config.max_len]
    else:
        if len(inputs) < config.max_len:
            pad = [tokenizer.pad_token_id] * (config.max_len - len(inputs))
            inputs = np.concatenate([inputs, pad])
        else:
            inputs = inputs[:config.max_len]

    return inputs


def preprocess_data(data_to_process, tokenizer, config):
    label_id= []
    label_ids = []
    dec_input_ids = []
    input_ids = []    

    for i in range(len(data_to_process['Text'])):
        input_ids.append(add_padding_data(tokenizer.encode(data_to_process['Text'][i], add_special_tokens=False), config, tokenizer, is_mlm=True))
        label_id.append(tokenizer.encode(data_to_process['Text'][i]))  
        label_id[i].append(tokenizer.eos_token_id) 
        dec_input_id = tokenizer('<s>')['input_ids']
        dec_input_id += label_id[i][:-1]
        dec_input_ids.append(add_padding_data(dec_input_id, config, tokenizer))
        label_ids.append(add_ignored_data(label_id[i], config, input_ids[i], tokenizer))
        
    return {'input_ids': torch.tensor(input_ids),
            'attention_mask' : torch.tensor((np.array(input_ids) != tokenizer.pad_token_id).astype(int)),
            'decoder_input_ids': torch.tensor(dec_input_ids),
            'decoder_attention_mask': torch.tensor((np.array(dec_input_ids) != tokenizer.pad_token_id).astype(int)),
            'labels': torch.tensor(label_ids)}
