import torch
import config
import numpy as np
from tqdm import tqdm
from model import Model
from torch.optim import Adam
import matplotlib.pyplot as plt
import torch.nn.functional as F
from dataset import get_dataloader


def eval():
    model = Model().to(config.device)
    model.load_state_dict(torch.load("./models/model.pkl"))
    model.eval()
    loss_list = []
    acc_list = []
    test_dataloader = get_dataloader()
    with torch.no_grad():
        for input, target in test_dataloader:
            input = input.to(config.device)
            target = target.to(config.device)
            output = model(input)
            loss = F.nll_loss(output, target)
            loss_list.append(loss.item())
            # 准确率
            pred = output.max(dim=-1)[-1]
            acc_list.append(pred.eq(target).cpu().float().mean())
        print("loss:{:.6f},acc:{}".format(np.mean(loss_list), np.mean(acc_list)))


if __name__ == '__main__':
    eval()



