import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from run_classifier import *
from config import TRAIN_CSV_PATH, TEST_CSV_PATH, TOKENIZED_TRAIN_CSV_PATH, VOCAB, max_seq_length, \
    train_batch_size, num_train_optimization_steps, VALIDATION_RATIO , RANDOM_STATE, label_list
from utils import record_log

train = pd.read_csv(TRAIN_CSV_PATH, index_col='id')
test = pd.read_csv(TEST_CSV_PATH, index_col='id')
cols = ['title1_zh',
        'title2_zh',
        'label']
train = train.loc[:, cols]
test = test.loc[:, cols]
train.fillna('UNKNOWN', inplace=True)
test.fillna('UNKNOWN', inplace=True)
train.head(3)

counter = Counter(train.label)
print(counter)


train, val = train_test_split(
    train,
    test_size=VALIDATION_RATIO,
    random_state=RANDOM_STATE
)


train_examples = [InputExample('train', row.title1_zh, row.title2_zh, row.label) for row in train.itertuples()]
val_examples = [InputExample('val', row.title1_zh, row.title2_zh, row.label) for row in val.itertuples()]
test_examples = [InputExample('test', row.title1_zh, row.title2_zh, 'unrelated') for row in test.itertuples()]

print(len(train_examples))

orginal_total = len(train_examples)
train_examples = train_examples[:int(orginal_total * 0.2)]

tokenizer = BertTokenizer.from_pretrained(VOCAB)
train_features = convert_examples_to_features(train_examples, label_list, max_seq_length, tokenizer, "classification")

record_log("Running training", len(train_examples), train_batch_size, num_train_optimization_steps)

all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)

train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=train_batch_size)



