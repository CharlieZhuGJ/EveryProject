# https://blog.csdn.net/qq_18822147/article/details/108723112
# https://my.oschina.net/u/4579644/blog/4352720
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


def one_dim_transit(x, dim):
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
    for i in range(10):
        sum_probs += norm_probs[i]
        if sum_probs >= u:
            return candidates[i]


def gibbs(x):
    dim = len(x)
    new = x[:]
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
    x = [uniform_gen(), uniform_gen()]
    res = [x]
    for i in range(num):
        res.append(gibbs(x))
    return res


res = gibbs_sampling(2000)
plt_scatter(res, path='gibbs.jpg')


