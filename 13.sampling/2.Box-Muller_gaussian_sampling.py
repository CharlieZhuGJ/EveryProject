import numpy as np
from pylab import *
import random
"""
https: // blog.csdn.net / bitcarmanlee / article / details / 82795137
利用Box - Muller变换生成高斯分布随机数的方法可以总结为以下步骤：
1.生成两个随机数U1, U2 ∼ U[0, 1]
2.令R = − 2;θ = 2πU
3.z0 = Rcos(θ), z1 = Rsinθ
"""


def Box_Muller_sample():
    x = np.random.rand()  # 两个均匀分布分别为x, y
    y = np.random.rand()
    R = np.sqrt(-2 * np.log(x))
    theta = 2 * np.pi * y
    z0 = R * np.cos(theta)
    z1 = R * np.sin(theta)
    return z0, z1


def get_n_samples(n=100000):
    res = []
    a = -4  # 范围下限
    b = 4  # 范围上限
    for i in range(n):
        x = sample()
        x = a + (b - a) * x
        res.append(x)
    y = np.reshape(res, (n,1))
    hist(y, density=1, fc='c')  # 直方图

    x = arange(a, b, 0.1)
    plot(x, 1 / np.sqrt(2 * np.pi) * np.exp(-0.5 * x ** 2), 'g', lw=6)  # 标准正态分布
    xlabel('x', fontsize=24)
    ylabel('p(x)', fontsize=24)

    show()


get_n_samples()


