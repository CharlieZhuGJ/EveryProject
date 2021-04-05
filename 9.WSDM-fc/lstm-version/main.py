import pickle
from tqdm import tqdm
from dataset import MyDataset
from word_sequence import WordSequence

if __name__ == '__main__':
    print("start")
    ws = WordSequence()
    train_data = MyDataset(train=True)

    for idx in range(len(train_data)):
        if idx % 10000 == 0:
            print(idx)
        lines, labels = train_data[idx]
        ws.fit(lines)

    test_data = MyDataset(train=False)
    for idx in range(len(test_data)):
        if idx % 10000 == 0:
            print(idx)
        lines, labels = test_data[idx]
        ws.fit(lines)

    print("正在建立...")
    ws.build_vocab()
    print(len(ws))
    pickle.dump(ws, open("ws.pkl", "wb"))

#  为啥使用tqdm会多取元素？
# for lines, labels in tqdm(train_data, total=len(train_data)):
#     # 当您将total用作参数提供时tqdm，您可以估算出代码应运行多少次迭代，
#     # 因此它将为您提供预测信息（即使您提供的可迭代项没有长度）。
#     ws.fit(lines)

