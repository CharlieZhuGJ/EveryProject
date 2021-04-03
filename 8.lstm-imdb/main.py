import pickle
from tqdm import tqdm
from dataset import get_dataloader, ImdbDataset
from word_sequence import WordSequence

if __name__ == '__main__':
    print("start")
    ws = WordSequence()
    # train_data = get_dataloader(True)
    # test_data = get_dataloader(False)
    train_data = ImdbDataset(train=True)
    test_data = ImdbDataset(train=False)
    for reviews, labels in tqdm(train_data, total=len(train_data)):
        ws.fit(reviews)
        # for review in reviews:
        #     ws.fit(reviews)
    for reviews, labels in tqdm(test_data, total=len(test_data)):
        ws.fit(reviews)
        # for review in reviews:
        #     ws.fit(review)
    print("正在建立...")
    ws.build_vocab()
    print(len(ws))
    pickle.dump(ws, open("ws.pkl", "wb"))
