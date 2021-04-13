# https://blog.csdn.net/Dark_Scope/article/details/78937731
# https://my.oschina.net/u/4579644/blog/4352720
# https://zhuanlan.zhihu.com/p/146020807
"""

"""
from __future__ import division

import numpy as np
import matplotlib.pylab as plt

"""
目标矩阵P可以通过任意一个马尔科夫链状态转移矩阵Q以一定接受概率获得。
M-H采样算法过程如下：
　　　　1）输入我们任意选定的马尔科夫链状态转移矩阵Q，平稳分布π(x)，设定状态转移次数阈值n1，需要的样本个数n2
　　　　2）从任意简单概率分布采样得到初始状态值x0
　　　　3）for t=0 to n1+n2−1: 

　　　　　　a) 从条件概率分布Q(x|xt)中采样得到样本x∗
　　　　　　b) 从均匀分布采样u∼uniform[0,1]
　　　　　　c) 如果u<α(xt,x∗)=min{π(j)Q(j,i)π(i)Q(i,j),1}, 则接受转移xt→x∗，即xt+1=x∗
　　　　　　d) 否则不接受转移，即xt+1=xt
　　　　样本集(xn1,xn1+1,...,xn1+n2−1)即为我们需要的平稳分布对应的样本集。
"""
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


def mh_sample(n=20000):
    """
    使用metropolis Hastings生成样本
    :param n:
    :return:
    """
    sample = np.zeros(n)
    sample[0] = 0.5  # 初始化
    for i in range(n - 1):
        qs = qsample()  # 从转移矩阵Q(x)得到样本xt
        u = np.random.rand()  # 均匀分布
        # i=sample[i]是当前样本；j=qs是下一个样本
        p_j = p(qs)  # 下个样本的概率
        q_j_i = q(sample[i])  # 从当前样本生成下个样本的转移概率
        p_i = p(sample[i])   # 当前样本的概率
        q_i_j = q(qs)  # 下个样本转移到当前样本的概率
        alpha_ij = (p_j * q_j_i) / (p_i * q_i_j)  # alpha(i, j)表达式，这个改变解决了接受率低的问题

        if u < min(alpha_ij, 1):
            sample[i + 1] = qs  # 接受
        else:
            sample[i + 1] = sample[i]  # 拒绝

    return sample


x = np.arange(0, 4, 0.1)
real_data = p(x)
sample_data = mh_sample()
plt.plot(x, real_data, 'g', lw=3)  # 理想数据
plt.plot(x, q(x), 'r')  # Q(x)转移矩阵的数据
plt.hist(sample_data, bins=x, normed=1, fc='c')  # 采样生成的数据
plt.show()
