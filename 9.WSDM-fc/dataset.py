import os
import torch
import config
import numpy as np
import pandas as pd
from utils import jieba_tokenizer, en_tokenlizer
from torch.utils.data import DataLoader, Dataset

label_to_index = {
    'unrelated': 0,
    'agreed': 1,
    'disagreed': 2
}


class MyDataset(Dataset):
    def __init__(self, train=True):
        super(MyDataset, self).__init__()
        data_path = r"D:\Projects\EveryProject\9.WSDM-fc\data"
        if train:
            self.tokenized_data_path = data_path + r'.\tokenized_train.csv'
        else:
            self.tokenized_data_path = data_path + r'.\tokenized_test.csv'
        self.tokenized_data = pd.read_csv(self.tokenized_data_path)

        if train:
            self.label = self.tokenized_data.label
        else:
            self.label = None
        self.corpus_x1 = self.tokenized_data.title1_tokenized
        self.corpus_x2 = self.tokenized_data.title2_tokenized
        corpus = pd.concat([self.corpus_x1, self.corpus_x2])

    def __getitem__(self, idx):
        if idx >= len(self.tokenized_data):
            return [], None
        else:
            line_1 = self.corpus_x1[idx]
            if not isinstance(line_1, str):
                line_1 = str(line_1)
            line_1 = line_1.split(" ")

            line_2 = self.corpus_x2[idx]
            if not isinstance(line_2, str):
                line_2 = str(line_2)
            line_2 = line_2.split(" ")
            line = line_1 + line_2
            if self.label is None:
                label = self.label
            else:
                label = self.label[idx]
            return line, label

    def __len__(self):
        return len(self.tokenized_data)


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='float')[y]


def collate_fn(batch):
    """
    对batch数据进行处理
    :param batch:
    :return:
    """
    reviews, labels = zip(*batch)

    text_to_seq = [config.ws.transform(i, max_len=config.max_len) for i in reviews]
    reviews = torch.tensor(text_to_seq)

    labels = [label_to_index[x] for x in labels]
    # labels = to_categorical(labels, config.NUM_CLASSES)
    labels = torch.tensor(labels, dtype=torch.long)

    return reviews, labels


def get_dataloader(train):
    dataset = MyDataset(train=True)
    batch_size = config.train_batch_size if train else config.test_batch_size
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    return data_loader


if __name__ == '__main__':
    # dataset = MyDataset(train=True)
    # print(dataset[1])

    batched_data = get_dataloader(train=True)
    for idx, (title, label) in enumerate(batched_data):
        print(idx)
        print(title)
        print(label)
        # break
