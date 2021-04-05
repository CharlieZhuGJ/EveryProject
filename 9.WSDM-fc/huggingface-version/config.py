import os
import sys
import torch
import zipfile
import datetime
import numpy as np
import pandas as pd
from dataset import train_examples

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()

gradient_accumulation_steps = 1
train_batch_size = 32
eval_batch_size = 128

train_batch_size = train_batch_size // gradient_accumulation_steps

output_dir = "output"
bert_model = 'bert-base-chinese'
num_train_epochs = 3
num_train_optimization_steps = int(len(train_examples) / train_batch_size / gradient_accumulation_steps) * num_train_epochs

cache_dir = "model"
learning_rate = 5e-5
warmup_proportion = 0.1
max_seq_length = 128
label_list = ['unrelated', 'agreed', 'disagreed']


VOCAB = './bert-base-chinese/vocab.txt'
MODEL = './bert-base-chinese'

VALIDATION_RATIO = 0.1
RANDOM_STATE = 9527

global_step = 0
nb_tr_steps = 0
tr_loss = 0

CONFIG_NAME = "fc_config.json"
WEIGHTS_NAME = "fc_pytorch_model.bin"


output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
output_config_file = os.path.join(output_dir, CONFIG_NAME)
output_eval_file = os.path.join(output_dir, "eval_results.txt")

if os.path.isdir("./fake-news-pair-classification-challenge"):
    TRAIN_CSV_PATH = './fake-news-pair-classification-challenge/train.csv'
    TEST_CSV_PATH = './fake-news-pair-classification-challenge/test.csv'
    TOKENIZED_TRAIN_CSV_PATH = "./fake-news-pair-classification-challenge/tokenized_train.csv"
else:
    TRAIN_CSV_PATH = '../input/train.csv'
    TEST_CSV_PATH = '../input/test.csv'
    TOKENIZED_TRAIN_CSV_PATH = ""


