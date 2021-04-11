# https://blog.csdn.net/qq_18822147/article/details/108723112
# https://blog.csdn.net/bitcarmanlee/article/details/82795137

from __future__ import division

import numpy as np
import matplotlib.pylab as plt

mu = 3
sigma = 10


def q(x):
    """
    转移矩阵Q,因为是模拟数字，只有一维，所以Q是个数字(1*1)
    :param x:
    :return:
    """
    return np.exp(-(x - mu) ** 2 / (sigma ** 2))


def qsample():
    """
   按照转移矩阵Q生成样本
    :return:
    """
    return np.random.normal(mu, sigma)


def p(x):
    """
    目标分布函数p(x)
    :param x:
    :return:
    """
    return 0.3 * np.exp(-(x - 0.3) ** 2) + 0.7 * np.exp(-(x - 2.) ** 2 / 0.3)


def mcmcsample(n=20000):
    """
    使用mcmc生成样本
    :param n:
    :return:
    """
    sample = np.zeros(n)
    sample[0] = 0.5  # 初始化
    for i in range(n - 1):
        qs = qsample()  # 从转移矩阵Q(x)得到样本xt
        u = np.random.rand()  # 均匀分布
        alpha_i_j = (p(qs) * q(sample[i])) / (p(sample[i]) * qs)  # alpha(i, j)表达式
        if u < min(alpha_i_j, 1):
            sample[i + 1] = qs  # 接受
        else:
            sample[i + 1] = sample[i]  # 拒绝

    return sample


x = np.arange(0, 4, 0.1)
realdata = p(x)
sampledata = mcmcsample()
plt.plot(x, realdata, 'g', lw=3)  # 理想数据
plt.plot(x, q(x), 'r')  # Q(x)转移矩阵的数据
plt.hist(sampledata, bins=x, normed=1, fc='c')  # 采样生成的数据
plt.show()




