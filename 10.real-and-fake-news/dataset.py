import os
import torch
import config
import numpy as np
import pandas as pd
from utils import jieba_tokenizer, en_tokenlizer
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataset import random_split

label_to_index = {
    'REAL': 0,
    'FAKE': 1
}


class MyDataset(Dataset):
    def __init__(self):
        super(MyDataset, self).__init__()
        data_path = r"D:\Projects\EveryProject\10.real-and-fake-news\news.csv"
        self.data = pd.read_csv(data_path)

        self.title = self.data.title
        self.text = self.data.text
        self.label = self.data.label

    def __getitem__(self, idx):
        if idx >= len(self.data):
            return [], None
        else:
            title = en_tokenlizer(self.title[idx])
            text = en_tokenlizer(self.text[idx])
            label = self.label[idx]
            return title, text, label

    def __len__(self):
        return len(self.data)


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='float')[y]


def collate_fn(batch):
    """
    对batch数据进行处理
    :param batch:
    :return:
    """
    titles, texts, labels = zip(*batch)
    combined_texts = [ab[0] + ab[1] for ab in zip(titles, texts)]
    text_to_seq = [config.ws.transform(sent, max_len=config.MAX_SEQUENCE_LENGTH) for sent in combined_texts]
    texts = torch.tensor(text_to_seq)

    labels = [label_to_index[x] for x in labels]
    labels = torch.tensor(labels, dtype=torch.long)

    return texts, labels


def get_dataloader():
    dataset = MyDataset()

    num_train = int(len(dataset) * 0.7)
    train_dataset, test_dataset = random_split(dataset, [num_train, len(dataset) - num_train])

    train_data_loader = DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True, collate_fn=collate_fn)
    test_data_loader = DataLoader(test_dataset, batch_size=config.test_batch_size, shuffle=True, collate_fn=collate_fn)

    return train_data_loader, test_data_loader


if __name__ == '__main__':
    train_, test_, = get_dataloader()
    for idx, (title, label) in enumerate(train_):
        print(idx)
        print(title)
        print(label)
        # break

    for idx, (title, label) in enumerate(test_):
        print(idx)
        print(title)
        print(label)
        # break

