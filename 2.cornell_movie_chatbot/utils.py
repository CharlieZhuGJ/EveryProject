import re
import os
import pickle
import torch
import jieba.posseg as pseg
from config import device, trimed_datafile, corpus, voc_file


# 5.定义训练步骤
def maskNLLLoss(inp, target, mask):
    """
    定义交叉熵损失函数
    :param inp:
    :param target:
    :param mask:
    :return:
    """
    nTotal = mask.sum()
    crossEntropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)).squeeze(1))
    loss = crossEntropy.masked_select(mask).mean()
    loss = loss.to(device)
    return loss, nTotal.item()


def en_tokenlizer(sentence):
    """
    进行文本分词
    :param sentence: str
    :return: [str,str,str]
    """

    fileters = ['!', '"', '#', '$', '%', '&', '\(', '\)', '\*', '\+', ',', '-', '\.', '/', ':', ';', '<', '=', '>',
                '\?', '@', '\[', '\\', '\]', '^', '_', '`', '\{', '\|', '\}', '~', '\t', '\n', '\x97', '\x96', '”',
                '“', ]
    sentence = sentence.lower()  # 把大写转化为小写
    sentence = re.sub("<br />", " ", sentence)
    # sentence = re.sub("I'm","I am",sentence)
    # sentence = re.sub("isn't","is not",sentence)
    sentence = re.sub("|".join(fileters), " ", sentence)
    result = [i for i in sentence.split(" ") if len(i) > 0]

    return result


def jieba_tokenizer(text):
    """
    使用结巴进行中文分词
    :param text:
    :return:
    """
    words = pseg.cut(text)
    res = [word for word, flag in words if flag != 'x']
    return res


def get_saved_voc():
    voc = pickle.load(open(os.path.join(corpus, voc_file), 'rb'))
    return voc


def get_saved_paird():
    pairs = []
    with open(os.path.join(corpus, trimed_datafile), 'r') as txt:
        lines = txt.readlines()
        for line in lines:
            ab = line.split("\t")
            first = ab[0].strip()
            second = ab[1].strip()
            pairs.append([first, second])
    return pairs
