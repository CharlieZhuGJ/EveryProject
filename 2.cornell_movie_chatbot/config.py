import os
import torch
import codecs

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

####################################################################
# dataset.py 相关参数
####################################################################
corpus_name = "cornell movie-dialogs corpus"  # 词汇文件夹名
corpus = os.path.join("data", corpus_name)  # 词汇文件夹路径
conv_filepath = os.path.join(corpus, "movie_conversations.txt")  # movie_conversations文件路径
movie_lines_file = os.path.join(corpus, "movie_lines.txt")  # movie_lines文件路径
datafile = os.path.join(corpus, "formatted_movie_lines.txt")  # 格式化电影对话数据文件路径，一行中有两句话，中间用\t隔开
save_dir = os.path.join("data", "save")  # 保存文件夹路径

trimed_datafile = "trimed_formatted_movie_lines.txt"  # 修剪过的格式化电影对话数据文件

delimiter = '\t'
delimiter = str(codecs.decode(delimiter, "unicode_escape"))  # 分隔符

MOVIE_LINES_FIELDS = ["lineID", "characterID", "movieID", "character", "text"]  # 电影字段信息
MOVIE_CONVERSATIONS_FIELDS = ["character1ID", "character2ID", "movieID", "utteranceIDs"]  # 对话字段信息

voc_file = "voc.pkl"

# 默认词向量
PAD_token = 0  # 填充符号
SOS_token = 1  # 句首符号
EOS_token = 2  # 句尾符号
MAX_LENGTH = 10  # 句子最大长度
MIN_COUNT = 3  # 修剪的最小字数阈值
small_batch_size = 5  # 每批数据数量


####################################################################
# train.py 相关参数
####################################################################
# 配置模型
model_name = 'cb_model'  # 模型名称
attn_model = 'dot'  # 注意力机制类型
# attn_model = 'general'
# attn_model = 'concat'
hidden_size = 500  # GRU网络中的输出维度，GRU网络中的第二个输入
embedding_dim = 500  # 输入单词的特征维数，GRU网络中的第一个输入
encoder_n_layers = 2  # encoder中网络层数
decoder_n_layers = 2  # decoder中网络层数
dropout = 0.1  # 神经节点丢弃率
batch_size = 64  # 批大小

# 设置检查点以加载; 如果从头开始，则设置为None
# loadFilename = None
checkpoint_iter = 4000
loadFilename = os.path.join(save_dir, model_name, corpus_name,
                            '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size),
                            '{}_checkpoint.tar'.format(checkpoint_iter))

# 配置训练/优化
clip = 50.0
teacher_forcing_ratio = 1.0  # 教师强迫率
learning_rate = 0.0001  # encoder中Adam优化器的学习率
decoder_learning_ratio = 5.0  # decoder中Adam优化器的学习率
n_iteration = 4000  # 迭代次数，取多少批数据
print_every = 1  # 每次打印多少
save_every = 500  # 每次保存多少


