import torch.nn as nn
import config
import torch.nn.functional as F


class Model(nn.Module):
    """
    线性模型，线性层最大20个单词（config.max_len），每个单词300个维度（NUM_EMBEDDING_DIM）
    """
    def __init__(self):
        super(Model, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=len(config.ws),
                                      embedding_dim=config.NUM_EMBEDDING_DIM,
                                      padding_idx=config.ws.PAD)
        self.fc = nn.Linear(config.max_len * config.NUM_EMBEDDING_DIM, config.NUM_CLASSES)

    def forward(self, input):
        input_embeded = self.embedding(input)
        input_embeded_viewed = input_embeded.view(input_embeded.size(0), -1)
        out = self.fc(input_embeded_viewed)
        res = F.log_softmax(out, dim=-1)
        return res
