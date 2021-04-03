import torch.nn as nn
import config
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=len(config.ws),
                                      embedding_dim=300,
                                      padding_idx=config.ws.PAD)
        self.fc = nn.Linear(config.max_len * 300, 2)

    def forward(self, input):
        input_embeded = self.embedding(input)
        input_embeded_viewed = input_embeded.view(input_embeded.size(0), -1)
        out = self.fc(input_embeded_viewed)
        res = F.log_softmax(out, dim=-1)
        return res
