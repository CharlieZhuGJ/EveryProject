import os
import torch
import pickle
from word_sequence import WordSequence
train_batch_size = 256
test_batch_size = 256

ws_model_file = 'ws.pkl'
if os.path.isfile(ws_model_file):
    ws = pickle.load(open(ws_model_file, 'rb'))
else:
    ws = WordSequence()
    print("生成字典")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

max_len = 20
hidden_size = 128
num_layers = 2
bidirectional = True
dropout = 0.4

# 分类数量
NUM_CLASSES = 3

# 语料库词汇大小
MAX_NUM_WORDS = 10000

# 标题最大长度
MAX_SEQUENCE_LENGTH = 20

# 每个单词的向量维度
NUM_EMBEDDING_DIM = 300

# LSTM 隐层维度
NUM_LSTM_UNITS = 128


