import torch
import config
import numpy as np
# from eval import eval
from tqdm import tqdm
from model import Model
from torch.optim import Adam
import torch.nn.functional as F
import matplotlib.pyplot as plt
from dataset import get_dataloader

model = Model().to(config.device)
optimizer = Adam(model.parameters(), lr=config.LR)
train_loss_list = []
train_acc_list = []
test_loss_list = []
test_acc_list = []

train_dataloader, test_dataloader = get_dataloader()


def train(epoch):
    bar = tqdm(train_dataloader, total=len(train_dataloader))
    loss_list = []
    acc_list = []
    for idx, (inp, target) in enumerate(bar):
        optimizer.zero_grad()
        inp = inp.to(config.device)
        target = target.to(config.device)
        output = model(inp)
        loss = F.nll_loss(output, target)
        loss.backward()
        loss_list.append(loss.item())
        # 准确率
        pred = output.max(dim=-1)[-1]
        acc_list.append(pred.eq(target).cpu().float().mean())

        optimizer.step()
        bar.set_description("epoch:{} idx:{} loss:{:.6f} acc:{:.6f}".format(epoch, idx, np.mean(loss_list), np.mean(acc_list)))

        if idx % 10 == 0:
            torch.save(model.state_dict(), "./models/model.pkl")
            torch.save(optimizer.state_dict(), "./models/optimizer.pkl")

        train_loss_list.append(loss_list)
        train_acc_list.append(acc_list)


def eval():
    model = Model().to(config.device)
    model.load_state_dict(torch.load("./models/model.pkl"))
    model.eval()
    loss_list = []
    acc_list = []
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
    test_loss_list.append(loss_list)
    test_acc_list.append(acc_list)


def plot_and_save_png(train_loss_or_acc,test_loss_or_acc,  name="loss_or_acc"):
    plt.figure(figsize=(10, 8))
    plt.plot(range(len(train_loss_or_acc[-1])), train_loss_or_acc[-1],  color='red', linestyle='--', label='train')
    plt.plot(range(len(test_loss_or_acc[-1])), test_loss_or_acc[-1], color='green', linestyle='-', label='test')
    plt.legend()
    plt.savefig(name + ".png")
    plt.show()



if __name__ == "__main__":
    for i in range(5):
        train(i)
        print()
        eval()
    print("绘图并保存图片")
    plot_and_save_png(train_loss_list, test_loss_list, "loss")
    plot_and_save_png(train_acc_list, test_acc_list, "acc")

    print("结束")
