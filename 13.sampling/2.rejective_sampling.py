# https://blog.csdn.net/bitcarmanlee/article/details/82795137
# https://my.oschina.net/u/4579644/blog/4352720
import numpy as np
import matplotlib.pyplot as plt

"""
1.首先，确定常量k，使得p(x)总在kq(x)的下方。
2.x轴的方向：从q(x)分布抽样取得a。但是a不一定留下，会有一定的几率被拒绝。
3.y轴的方向：从均匀分布(0, kq(a))中抽样得到u。如果u>p(a)，也就是落到了灰色的区域中，拒绝，否则接受这次抽样。

在高维的情况下，Rejection Sampling有两个问题：
1.合适的q分布很难找
2.很难确定一个合理的k值，导致拒绝率很高。
"""


def qsample():
    """
    使用均匀分布作为q(x)，返回采样
    :return:
    """
    return np.random.rand() * 4.


def p(x):
    """
    目标分布
    :param x:
    :return:
    """
    return 0.3 * np.exp(-(x - 0.3) ** 2) + 0.7 * np.exp(-(x - 2.) ** 2 / 0.3)


def rejection(nsamples):
    """
    拒绝采样
    :param nsamples:
    :return:
    """
    M = 0.72  # 0.8 k值
    samples = np.zeros(nsamples, dtype=float)
    count = 0
    for i in range(nsamples):
        accept = False
        while not accept:
            # x方向上进行采样
            x = qsample()
            u = np.random.rand() * M
            # y方向上，如果采样值小于p(x)，表明在p分布内，则接受；否则拒绝。
            if u < p(x):
                accept = True
                samples[i] = x
            else:
                count += 1
    print("reject count: ", count)
    return samples


x = np.arange(0, 4, 0.01)
x2 = np.arange(-0.5, 4.5, 0.1)
realdata = 0.3 * np.exp(-(x - 0.3) ** 2) + 0.7 * np.exp(-(x - 2.) ** 2 / 0.3)
box = np.ones(len(x2)) * 0.75  # 0.8
box[:5] = 0
box[-5:] = 0
plt.plot(x, realdata, 'g', lw=3)
plt.plot(x2, box, 'r--', lw=3)

import time

t0 = time.time()
samples = rejection(10000)
t1 = time.time()
print("Time ", t1 - t0)

plt.hist(samples, 15, normed=1, fc='c', color='r')
plt.xlabel('x', fontsize=24)
plt.ylabel('p(x)', fontsize=24)
plt.axis([-0.5, 4.5, 0, 1])
plt.show()
