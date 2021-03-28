import os
import csv
import codecs
from io import open

corpus_name = "cornell movie-dialogs corpus"
corpus = os.path.join("data", corpus_name)


# 2.1 加载数据并转换为对话格式
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
    sentence_pairs = extract_sentence_pairs(conversations)
    for pair in sentence_pairs:
        writer.writerow(pair)

# 打印一个样本的行
print("\nSample lines from file:")
print_file_by_line(datafile)


