# https://blog.csdn.net/Dark_Scope/article/details/78937731
# https://my.oschina.net/u/4579644/blog/4352720
# https://zhuanlan.zhihu.com/p/146020807
"""

"""
import numpy as np
import matplotlib.pyplot as plt


def p(x):
    """
    计算定义域一个随机值
    :return:
    """
    return 1 / (2 * np.pi) * np.exp(-x[0] ** 2 - x[1] ** 2)


def uniform_gen():
    """
    计算定义域[-2,2]一个随机值
    :return:
    """
    return np.random.random() * 4 - 2


def metropolis(x0):
    """
    通过metropolis-Hastings采样判断是否保留样本
    :param x0:
    :return:
    """
    x = (uniform_gen(), uniform_gen())
    u = np.random.uniform()
    alpha = min(1, p(x) / p(x0))
    if u < alpha:
        return x
    else:
        return x0


def gen_px(num):
    x = (uniform_gen(), uniform_gen())
    res = [x]
    for i in range(num):
        x = metropolis(x)
        res.append(x)
    return np.array(res)


def plt_scatter(x, path=None):
    x = np.array(x)
    plt.scatter(x[:, 0], x[:, 1])
    if path:
        plt.savefig(path)
    plt.show()


res = gen_px(5000)
plt_scatter(res, 'MCMC.jpg')
