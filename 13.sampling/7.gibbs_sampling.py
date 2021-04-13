# https://blog.csdn.net/qq_18822147/article/details/108723112
# https://my.oschina.net/u/4579644/blog/4352720
import numpy as np
import matplotlib.pyplot as plt

"""
比如一个n维的概率分布π(x1,x2,...xn)，我们可以通过在n个坐标轴上轮换采样，来得到新的样本。
对于轮换到的任意一个坐标轴xi上的转移，马尔科夫链的状态转移概率为P(xi|x1,x2,...,xi−1,xi+1,...,xn)，
即固定n−1个坐标轴，在某一个坐标轴上移动。

具体的算法过程如下：
　　　　1）输入平稳分布π(x1,x2，...,xn)或者对应的所有特征的条件概率分布，设定状态转移次数阈值n1，需要的样本个数n2
　　　　2）随机初始化初始状态值(x(0)1,x(0)2,...,x(0)n
　　　　3）for t=0 to n1+n2−1: 
　　　　　　a) 从条件概率分布P(x1|x(t)2,x(t)3,...,x(t)n)中采样得到样本xt+11
　　　　　　b) 从条件概率分布P(x2|x(t+1)1,x(t)3,x(t)4,...,x(t)n)中采样得到样本xt+12
　　　　　　c)...
　　　　　　d) 从条件概率分布P(xj|x(t+1)1,x(t+1)2,...,x(t+1)j−1,x(t)j+1...,x(t)n)中采样得到样本xt+1j
　　　　　　e)...
　　　　　　f) 从条件概率分布P(xn|x(t+1)1,x(t+1)2,...,x(t+1)n−1)中采样得到样本xt+1n
　　　　样本集{(x(n1)1,x(n1)2,...,x(n1)n),...,(x(n1+n2−1)1,x(n1+n2−1)2,...,x(n1+n2−1)n)}即为我们需要的平稳分布对应的样本集。
"""


def p(x):
    """
    目标分布函数，生成给定样本的概率。
    :return:
    """
    return 1 / (2 * np.pi) * np.exp(-x[0] ** 2 - x[1] ** 2)


def uniform_gen():
    """
    计算定义域[-2,2]一个随机值
    :return:
    """
    return np.random.random() * 4 - 2


def one_dim_transit(x, dim):
    """
    坐标转移生成某一维度的样本
    :param x:
    :param dim:
    :return:
    """
    candidates = []
    probs = []
    for i in range(10):
        tmp = x[:]
        tmp[dim] = uniform_gen()
        candidates.append(tmp[dim])
        probs.append(p(tmp))
    norm_probs = np.array(probs) / sum(probs)
    u = np.random.uniform()
    sum_probs = 0
    # 累计概率超过阈值，则接受该样本；由于累积概率之和为1，一定能产生样本
    for i in range(10):
        sum_probs += norm_probs[i]
        if sum_probs >= u:
            return candidates[i]


def gibbs(x):
    dim = len(x)  # 样本维度
    new = x[:]  # 复制初始样本，生成新变量集合
    for i in range(dim):
        new[i] = one_dim_transit(new, i)
    return new


def plt_scatter(x, path=None):
    x = np.array(x)
    plt.scatter(x[:, 0], x[:, 1])
    if path:
        plt.savefig(path)
    plt.show()


def gibbs_sampling(num=1000):
    # 生成初始样本
    x = [uniform_gen(), uniform_gen()]
    res = [x]
    for i in range(num):
        res.append(gibbs(x))
    return res


res = gibbs_sampling(2000)
plt_scatter(res, path='gibbs.jpg')
