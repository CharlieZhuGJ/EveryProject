class WordSequence():
    UNK_TAG = "<UNK>"
    PAD_TAG = "<PAD>"
    UNK = 1
    PAD = 0

    def __init__(self):
        self.dict = {
            self.UNK_TAG: self.UNK,
            self.PAD_TAG: self.PAD
        }
        self.count = {}
        self.inverse_dict = {}

    def fit(self, sentence):
        """
        统计词频
        :param sentence:
        :return:
        """
        for word in sentence:
            self.count[word] = self.count.get(word, 0) + 1

    def build_vocab(self, min_count=0, max_count=None, max_num_words=None):
        """
        根据条件构建 词典
        :param min_count:最小词频
        :param max_count: 最大词频
        :param max_num_words: 最大词语数
        :return:
        """
        if min_count is not None:
            self.count = {word: count for word, count in self.count.items() if count > min_count}
        if max_count is not None:
            self.count = {word: count for word, count in self.count.items() if count < max_count}
        if max_num_words is not None:
            # 排序
            self.count = dict(sorted(self.count.items(), key=lambda x: x[-1], reverse=True)[:max_num_words])

        for word in self.count:
            # 每次word对应一个数字
            self.dict[word] = len(self.dict)
            # 把dict进行翻转
            self.inverse_dict = dict(zip(self.dict.values(), self.dict.keys()))

    def transform(self, sentence, max_len=None):
        """
        把句子转化为数字序列
        :param sentence:
        :return:
        """
        if len(sentence) > max_len:
            sentence = sentence[:max_len]
        else:
            sentence = sentence + [self.PAD_TAG] * (max_len - len(sentence))
        return [self.dict.get(i, 1) for i in sentence]

    def inverse_transform(self, indexs):
        """
        把数字序列转化为字符
        :param indexs: 句子的索引列表
        :return:
        """
        return [self.inverse_dict.get(i, "<UNK>") for i in indexs]

    def __len__(self):
        return len(self.dict)


if __name__ == '__main__':
    sentences = [["今天", "天气", "很", "好"],
                 ["今天", "去", "吃", "什么"]]

    ws = WordSequence()
    for sentence in sentences:
        ws.fit(sentence)

    ws.build_vocab(min_count=0)
    print(ws.dict)
    ret = ws.transform(["今天", "好热", "呀", "呀", "呀", "呀", "呀", "呀", "呀"], max_len=5)
    print(ret)
    ret = ws.inverse_transform(ret)
    print(ret)
