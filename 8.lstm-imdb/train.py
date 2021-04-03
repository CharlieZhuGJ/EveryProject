import torch
import config
import numpy as np
from eval import eval
from tqdm import tqdm
from model import Model
from torch.optim import Adam
import torch.nn.functional as F
import matplotlib.pyplot as plt
from dataset import get_dataloader

model = Model().to(config.device)
optimizer = Adam(model.parameters(), lr=0.001)
loss_list = []


def train(epoch):
    train_dataloader = get_dataloader(train=True)
    bar = tqdm(train_dataloader, total=len(train_dataloader))

    for idx, (inp, target) in enumerate(bar):
        optimizer.zero_grad()
        inp = inp.to()
        target = target.to()

        output = model(inp)
        loss = F.nll_loss(output, target)
        loss.backward()
        loss_list.append(loss.item())

        optimizer.step()
        bar.set_description("epoch:{} idx:{} loss:{:.6f}".format(epoch, idx, np.mean(loss_list)))

        if idx % 10 == 0:
            torch.save(model.state_dict(), "./models/model.pkl")
            torch.save(optimizer.state_dict(), "./models/optimizer.pkl")


if __name__ == "__main__":
    for i in range(5):
        train(i)
        eval()
    plt.figure(figsize=(20, 8))
    plt.plot(range(len(loss_list)), loss_list)
