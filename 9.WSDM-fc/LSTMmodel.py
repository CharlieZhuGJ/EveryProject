"""
构建模型
"""
import torch.nn as nn
import torch
import config
import torch.nn.functional as F


class ImdbModel(nn.Module):
    def __init__(self):
        super(ImdbModel, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=len(config.ws), embedding_dim=300, padding_idx=config.ws.PAD)
        self.lstm = nn.LSTM(input_size=200, hidden_size=64, num_layers=2, batch_first=True, bidirectional=True,
                            dropout=0.5)
        self.fc1 = nn.Linear(64 * 2, 64)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, input):
        """
        :param input:
        :return:
        """
        input_embeded = self.embedding(input)  # [batch_size,seq_len,200]

        output, (h_n, c_n) = self.lstm(input_embeded)
        out = torch.cat(h_n[-1, :, :], h_n[-2, :, :], dim=-1)  # 拼接正向最后一个输出和反向最后一个输出

        # 进行全连接
        out_fc1 = self.fc1(out)
        # 进行relu
        out_fc1_relu = F.relu(out_fc1)
        # 全连接
        out = self.fc2(out_fc1_relu)
        res = F.log_softmax(out, dim=-1)
        return res

