import os
import sys
import torch
import zipfile
import datetime
import numpy as np
import pandas as pd
from config import *
from run_classifier import BertAdam, logger, convert_examples_to_features
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForSequenceClassification, BertForMaskedLM

tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

text = "[CLS] Who was Jim Henson? [SEP] Jim Henson was a puppeteer [SEP]"

tokenized_text = tokenizer.tokenize(text)
print(tokenized_text)

tokenizer = BertTokenizer.from_pretrained(VOCAB)
model = BertForSequenceClassification.from_pretrained(MODEL,
                                                      cache_dir=cache_dir,
                                                      num_labels=3)
model.to(device)

if n_gpu > 1:
    model = torch.nn.DataParallel(model)
print(model)
print(tokenizer)

# Prepare optimizer
param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
optimizer = BertAdam(optimizer_grouped_parameters,
                     lr=learning_rate,
                     warmup=warmup_proportion,
                     t_total=num_train_optimization_steps)














