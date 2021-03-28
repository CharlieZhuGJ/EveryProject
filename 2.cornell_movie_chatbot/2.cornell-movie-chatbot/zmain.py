import os
import csv
import torch
import random
import codecs
import itertools
import unicodedata
from io import open
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

corpus_name = "cornell movie-dialogs corpus"
corpus = os.path.join("data", corpus_name)


def print_file_by_line(file, n=10):
    """
    打印文件前n行数据
    """
    with open(file, "rb") as txt:
        lines = txt.readlines()
    for line in lines[:n]:
        print(line)


movie_lines_file = os.path.join(corpus, "movie_lines.txt")
print_file_by_line(movie_lines_file)


def get_movie_data_dict(file, fields):
    """
    获取字典格式的对话数据
    @param file:
    @param fields:
    @return:
    """
    movie_data_dict = {}
    with open(file, 'r', encoding='iso-8859-1') as f:
        for line in f:
            values = line.split(" +++$+++ ")
            # Extract fields
            lineObj = {}
            for i, field in enumerate(fields):
                lineObj[field] = values[i]
            movie_data_dict[lineObj['lineID']] = lineObj
    return movie_data_dict


def get_conversations(file, lines, fields):
    """
    获取对话数据
    @param file:
    @param lines:
    @param fields:
    @return:
    """
    conversations = []
    with open(file, 'r', encoding='iso-8859-1') as f:
        for line in f:
            values = line.split(" +++$+++ ")
            # Extract fields
            convObj = {}
            for i, field in enumerate(fields):
                convObj[field] = values[i]
            # Convert string to list (convObj["utteranceIDs"] == "['L598485', 'L598486', ...]")
            lineIds = eval(convObj["utteranceIDs"])
            # Reassemble lines
            convObj["lines"] = []
            for lineId in lineIds:
                convObj["lines"].append(lines[lineId])
            conversations.append(convObj)
    return conversations


def extract_sentence_pairs(conversations):
    """
    从对话中提取句子对
    @param conversations:
    @return:
    """
    qa_pairs = []
    for conversation in conversations:
        # Iterate over all the lines of the conversation
        for i in range(len(conversation["lines"]) - 1):  # We ignore the last line (no answer for it)
            inputLine = conversation["lines"][i]["text"].strip()
            targetLine = conversation["lines"][i + 1]["text"].strip()
            # Filter wrong samples (if one of the lists is empty)
            if inputLine and targetLine:
                qa_pairs.append([inputLine, targetLine])
    return qa_pairs


# 定义新文件的路径
datafile = os.path.join(corpus, "formatted_movie_lines.txt")
delimiter = '\t'
delimiter = str(codecs.decode(delimiter, "unicode_escape"))

# 初始化行dict，对话列表和字段ID
lines = {}
conversations = []
MOVIE_LINES_FIELDS = ["lineID", "characterID", "movieID", "character", "text"]
MOVIE_CONVERSATIONS_FIELDS = ["character1ID", "character2ID", "movieID", "utteranceIDs"]

# 加载行和进程对话
print("\nProcessing corpus...")
lines = get_movie_data_dict(os.path.join(corpus, "movie_lines.txt"), MOVIE_LINES_FIELDS)
print("\nLoading conversations...")
conversations = get_conversations(os.path.join(corpus, "movie_conversations.txt"),
                                  lines, MOVIE_CONVERSATIONS_FIELDS)

# 写入新的csv文件
print("\nWriting newly formatted file...")
with open(datafile, 'w', encoding='utf-8') as output_file:
    writer = csv.writer(output_file, delimiter=delimiter, lineterminator='\n')
    for pair in extract_sentence_pairs(conversations):
        writer.writerow(pair)

# 打印一个样本的行
print("\nSample lines from file:")
print_file_by_line(datafile)

# 2.2 加载和清洗数据
# 默认词向量
PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token

# 2.2 加载和清洗数据
# 默认词向量
PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token


class Voc:
    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3  # Count SOS, EOS, PAD

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    # 删除低于特定计数阈值的单词
    def trim(self, min_count):
        if self.trimmed:
            return
        self.trimmed = True

        keep_words = []

        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print('keep_words {} / {} = {:.4f}'.format(
            len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)
        ))

        # 重初始化字典
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3  # Count default tokens

        for word in keep_words:
            self.addWord(word)


# 现在我们可以组装词汇表和查询/响应语句对。在使用数据之前，我们必须做一些预处理。
#
# 首先，我们必须使用unicodeToAscii将 unicode 字符串转换为 ASCII。
# 然后，我们应该将所有字母转换为小写字母并清洗掉除基本标点之 外的所有非字母字符 (normalizeString)。
# 最后，为了帮助训练收敛，我们将过滤掉长度大于MAX_LENGTH 的句子 (filterPairs)。

MAX_LENGTH = 10  # Maximum sentence length to consider


def unicode_to_ascii(s):
    """
    将Unicode字符串转换为纯ASCII，多亏了https://stackoverflow.com/a/518232/2809427
    @param s:
    @return:
    """
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def normalizeString(sent):
    """
    分三个步骤进行标准化：大小写转变（case folding）、词干提取（stemming）、词形还原（lemmatization）
    将所有字母转换为小写字母并清洗掉除基本标点之 外的所有非字母字符 (normalizeString)。
    @param sent:
    @return:
    """
    pass
    return sent


def get_Voc(data_file, corpus_name):
    """
    获取Voc对象，格式化对话数据并存放到列表中
    @param data_file:
    @param corpus_name:
    @return:
    """
    print("Reading lines...")
    # Read the file and split into lines
    lines = open(data_file, encoding='utf-8').read().strip().split('\n')
    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]
    voc = Voc(corpus_name)
    return voc, pairs


def filter_or_not_pairs(p):
    """
    如果对 'p' 中的两个句子都低于 MAX_LENGTH 阈值，则返回True
    @param p:
    @return:
    """
    # Input sequences need to preserve the last word for EOS token
    return len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH


def filter_pairs(p):
    """
    过滤满足条件的 pairs对话
    @param p:
    @return:
    """
    return [p for p in p if filter_or_not_pairs(pair)]


def get_prepared_data(corpus, corpus_name, datafile, save_dir):
    """
    使用上面定义的函数，返回一个填充的voc对象和对列表
    @param corpus:
    @param corpus_name:
    @param datafile:
    @param save_dir:
    @return:
    """
    print("Start preparing training data ...")
    voc, pairs = get_Voc(datafile, corpus_name)
    print("Read {!s} sentence pairs".format(len(pairs)))
    pairs = filter_pairs(pairs)
    print("Trimmed to {!s} sentence pairs".format(len(pairs)))
    print("Counting words...")
    for pair in pairs:
        voc.addSentence(pair[0])
        voc.addSentence(pair[1])
    print("Counted words:", voc.num_words)
    return voc, pairs


# 加载/组装voc和对
save_dir = os.path.join("data", "save")
voc, pairs = get_prepared_data(corpus, corpus_name, datafile, save_dir)
# 打印一些对进行验证
print("\n pairs:")
for pair in pairs[:10]:
    print(pair)

# 另一种有利于让训练更快收敛的策略是去除词汇表中很少使用的单词。减少特征空间也会降低模型学习目标函数的难度。
# 我们通过以下两个步 骤完成这个操作:
# 使用voc.trim函数去除 MIN_COUNT 阈值以下单词 。
# 如果句子中包含词频过小的单词，那么整个句子也被过滤掉。

MIN_COUNT = 3  # 修剪的最小字数阈值


def trim_rare_words(voc, pairs, MIN_COUNT):
    """
    修剪来自voc的MIN_COUNT下使用的单词
    @param voc:
    @param pairs:
    @param MIN_COUNT:
    @return:
    """
    voc.trim(MIN_COUNT)
    # Filter out pairs with trimmed words
    keep_pairs = []
    for pair in pairs:
        input_sentence = pair[0]
        output_sentence = pair[1]
        keep_input = True
        keep_output = True
        # 检查输入句子
        for word in input_sentence.split(' '):
            if word not in voc.word2index:
                keep_input = False
                break
        # 检查输出句子
        for word in output_sentence.split(' '):
            if word not in voc.word2index:
                keep_output = False
                break

        # 只保留输入或输出句子中不包含修剪单词的对
        if keep_input and keep_output:
            keep_pairs.append(pair)

    print("Trimmed from {} pairs to {}, {:.4f} of total".format(len(pairs), len(keep_pairs),
                                                                len(keep_pairs) / len(pairs)))
    return keep_pairs


# 修剪voc和对
pairs = trim_rare_words(voc, pairs, MIN_COUNT)


# 3.为模型准备数据
def indexes_from_sentence(voc, sentence):
    """
    获取句子中单词索引
    @param voc:
    @param sentence:
    @return:
    """
    return [voc.word2index[word] for word in sentence.split(' ')] + [EOS_token]


def zero_padding(l, fillvalue=PAD_token):
    """
    zip对数据进行合并了，相当于行列转置了
    @param l:
    @param fillvalue:
    @return:
    """
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))


def binaryMatrix(l, value=PAD_token):
    """
    记录 PAD_token的位置为0， 其他的为1
    @param l:
    @param value:
    @return:
    """
    m = []
    for i, seq in enumerate(l):
        m.append([])
        for token in seq:
            if token == PAD_token:
                m[i].append(0)
            else:
                m[i].append(1)
    return m


def inputVar(l, voc):
    """
    返回填充前（加入结束index EOS_token做标记）的长度 和 填充后的输入序列张量
    @param l:
    @param voc:
    @return:
    """
    indexes_batch = [indexes_from_sentence(voc, sentence) for sentence in l]
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    padList = zero_padding(indexes_batch)
    padVar = torch.LongTensor(padList)
    return padVar, lengths


def outputVar(l, voc):
    """
    返回填充前（加入结束index EOS_token做标记）最长的一个长度 和 填充后的输入序列张量, 和 填充后的标记 mask
    @param l:
    @param voc:
    @return:
    """
    indexes_batch = [indexes_from_sentence(voc, sentence) for sentence in l]
    max_target_len = max([len(indexes) for indexes in indexes_batch])
    padList = zero_padding(indexes_batch)
    mask = binaryMatrix(padList)
    mask = torch.ByteTensor(mask)
    padVar = torch.LongTensor(padList)
    return padVar, mask, max_target_len


def batch2TrainData(voc, pair_batch):
    """
    返回给定batch对的所有项目
    @param voc:
    @param pair_batch:
    @return:
    """
    pair_batch.sort(key=lambda x: len(x[0].split(" ")), reverse=True)
    input_batch, output_batch = [], []
    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])
    inp, lengths = inputVar(input_batch, voc)
    output, mask, max_target_len = outputVar(output_batch, voc)
    return inp, lengths, output, mask, max_target_len


# 验证例子
small_batch_size = 5
batches = batch2TrainData(voc, [random.choice(pairs) for _ in range(small_batch_size)])
input_variable, lengths, target_variable, mask, max_target_len = batches

print("input_variable:", input_variable)
print("lengths:", lengths)
print("target_variable:", target_variable)
print("mask:", mask)
print("max_target_len:", max_target_len)
