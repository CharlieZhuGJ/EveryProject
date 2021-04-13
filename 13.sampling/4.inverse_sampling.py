# https://baike.baidu.com/item/%E9%80%86%E5%8F%98%E6%8D%A2%E9%87%87%E6%A0%B7/22934385?fr=aladdin
import numpy as np
import random

"""
假设X为一个连续随机变量，其累积分布函数为F(x)。此时，随机变量Y=F(x)服从区间[0,1]上的均匀分布。
逆变换采样即是将该过程反过来进行：首先对于随机变量 Y，我们从0至1中随机均匀抽取一个数u。
之后，由于随机变量F_1(X)与X有着相同的分布，即可看作是从分布中生成的随机样本。
"""


def F(x):
    """
    变量的累积分布函数
    :param x:
    :return:
    """
    val = 1 - np.exp(-np.sqrt(x))
    return val


def F_1(x):
    """
    逆函数
    :param x:
    :return:
    """
    val = np.log(1 - x)
    return val ** 2


n_samples = 10  # 采样数量
us = []
samples = []  # 样本
for i in range(n_samples):
    u = np.random.rand()
    sample = F_1(u)
    us.append(u)
    samples.append(sample)

print(us)
print(samples)
