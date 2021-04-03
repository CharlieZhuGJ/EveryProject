import os
import torch
import pickle
from word_sequence import WordSequence
train_batch_size = 64
test_batch_size = 64

# ws = WordSequence()
ws_model_file = "ws.pkl"
if os.path.isfile(ws_model_file):
    ws = pickle.load(open('ws.pkl', 'rb'))
else:
    os.system("python main.py")
    print("生成字典")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

max_len = 200
hidden_size = 128
num_layers = 2
bidirectional = True
dropout = 0.4

