import os
import sys
import torch
import zipfile
import datetime
import numpy as np
import pandas as pd
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForSequenceClassification, BertForMaskedLM
from dataset import train_data, train_dataloader
from run_classifier import convert_examples_to_features
from model import model, optimizer
from config import device, n_gpu, output_dir, num_train_epochs, train_batch_size, \
    gradient_accumulation_steps, global_step, CONFIG_NAME, WEIGHTS_NAME, max_seq_length, label_list
import logging
from torch.utils.data.dataset import TensorDataset
from torch.utils.data import RandomSampler, SequentialSampler
from torch.utils.data.dataloader import DataLoader
from model import tokenizer
from dataset import test_examples, TEST_CSV_PATH

logger = logging.getLogger(__name__)
# step 1:
# run train.py to get model;
#
# step 2:
# run
#

def predict(model, tokenizer, examples, label_list, eval_batch_size=128):
    model.to(device)
    eval_examples = examples
    eval_features = convert_examples_to_features(eval_examples, label_list, max_seq_length, tokenizer, "classification")
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", eval_batch_size)
    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=eval_batch_size)

    model.eval()
    nb_eval_steps, nb_eval_examples = 0, 0

    res = []
    for input_ids, input_mask, segment_ids, label_ids in eval_dataloader:
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)

        with torch.no_grad():
            logits = model(input_ids, segment_ids, input_mask)
        logits = logits.detach().cpu().numpy()
        res.extend(logits.argmax(-1))
        nb_eval_steps += 1
    return res


res = predict(model, tokenizer, test_examples, label_list)
predict(model, tokenizer, test_examples[:10], label_list)

cat_map = {idx: lab for idx, lab in enumerate(label_list)}
res = [cat_map[c] for c in res]


test = pd.read_csv(TEST_CSV_PATH, index_col='id')

# For Submission
test['Category'] = res
submission = test.loc[:, ['Category']].reset_index()

submission.columns = ['Id', 'Category']
submission.to_csv('submission.csv', index=False)
submission.head()



