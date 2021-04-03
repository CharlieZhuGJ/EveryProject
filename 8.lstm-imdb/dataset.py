import os
import torch
import config
from utils import tokenlize
from torch.utils.data import DataLoader, Dataset


class ImdbDataset(Dataset):
    def __init__(self, train=True):
        super(ImdbDataset, self).__init__()
        data_path = r"D:\GitHub\lstm-sentimental-analysis\aclImdb_v1\aclImdb"
        data_path += r"\train" if train else r"\test"
        self.total_path = []
        for temp_path in [r"\pos", r"\neg"]:
            cur_path = data_path + temp_path
            self.total_path += [os.path.join(cur_path, i) for i in os.listdir(cur_path) if i.endswith(".txt")]

    def __getitem__(self, idx):
        file = self.total_path[idx]
        review = open(file, encoding="utf-8").read()
        review = tokenlize(review)
        label = int(file.split("_")[-1].split(".")[0])
        label = 0 if label < 5 else 1
        return review, label

    def __len__(self):
        return len(self.total_path)


def collate_fn(batch):
    """
    对batch数据进行处理
    :param batch:
    :return:
    """
    reviews, labels = zip(*batch)
    text_to_seq = [config.ws.transform(i, max_len=config.max_len) for i in reviews]
    reviews = torch.tensor(text_to_seq)
    labels = torch.tensor(labels)
    return reviews, labels


def get_dataloader(train):
    imdbdataset = ImdbDataset(train=True)
    batch_size = config.train_batch_size if train else config.test_batch_size
    data_loader = DataLoader(imdbdataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    return data_loader


if __name__ == '__main__':
    dataset = ImdbDataset(train=True)
    print(dataset[1])

    batched_data = get_dataloader(train=True)
    for idx, (review, label) in enumerate(batched_data):
        print(idx)
        print(review)
        print(label)
        # break


