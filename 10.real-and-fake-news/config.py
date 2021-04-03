import os
import torch
import pickle
from word_sequence import WordSequence
train_batch_size = 64
test_batch_size = 64

# 分类数量
NUM_CLASSES = 2
# 语料库词汇大小
MAX_NUM_WORDS = 10000
# 句子最大长度, 10.512707+765.578690
MAX_SEQUENCE_LENGTH = 800

# SGD优化函数的学习率
LR = 0.001


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






# 每个单词的向量维度
NUM_EMBEDDING_DIM = 300

# LSTM 隐层维度
NUM_LSTM_UNITS = 128


