# https://blog.csdn.net/bitcarmanlee/article/details/82795137
# 随机数生成（一）：均匀分布
# https://blog.csdn.net/JackyTintin/article/details/7798157
import random
import numpy as np


def get_random_num_by_linear_congruential_generator(seed, num, a=630360016, c=7, m=2 ** 31 - 1):
    # a 比例 c 常数 m 余数
    ans = np.zeros(num)
    n = 0
    x = seed
    while n < num:
        x = (a * x + c) % m
        y = x / m
        ans[n] = y
        n += 1
    return ans, x


ans, x = get_random_num_by_linear_congruential_generator(15, 10, a=630360016, c=7, m=2 ** 31 - 1)
print(ans)


# 线性同余法（Linear Congruential Generator, LCG）
# x(n+1) = (a*x(n) + b) mod m
# a、m为正整数，b为非负整数，a、b都小于m；X0为初始值（种子），是小于m的正整数。LCG产生的随机序列的周期最大为m，通常都会比m小。
# 序列具有最大的周期m，当且仅当：

# 计算圆周率


def linear_congruential_generator(x0, a, m, b):
    return (a * x0 + b) % m


def CalcPai(n):
    # 计算π值
    k = 0
    x = 0
    y = 0
    m = 99999999999
    a = 999
    b = 5
    for i in range(0, n):
        x = linear_congruential_generator(y, a, m, b)
        y = linear_congruential_generator(x, a, m, b)
        if x ** 2 + y ** 2 <= m ** 2:
            k = k + 1
    print(format(4 * k / n, '.2f'))


CalcPai(100000)

##############################################################################################

"""
验证rand的均匀分布
"""
from collections import defaultdict

dic = defaultdict(int)


def sample():
    u = np.random.rand()
    # 通过本函数可以返回一个或一组服从“0~1”均匀分布的随机样本值。随机样本取值范围是[0, 1)，不包括1。
    if u <= 0.1:
        dic["one"] += 1
    elif u <= 0.2:
        dic["two"] += 1
    elif u <= 0.3:
        dic["three"] += 1
    elif u <= 0.4:
        dic["four"] += 1
    elif u <= 0.5:
        dic["five"] += 1
    elif u <= 0.6:
        dic["six"] += 1
    elif u <= 0.7:
        dic["seven"] += 1
    elif u <= 0.8:
        dic["eight"] += 1
    elif u <= 0.9:
        dic["nine"] += 1
    else:
        dic["ten"] += 1


for i in range(10000):
    sample()
for k, v in dic.items():
    print(k, v)

"""
six 981
nine 993
two 1045
one 1018
five 971
eight 1023
four 1003
seven 1040
three 959
ten 967

通过结果可以知道，抽样生成的样本：符合均匀分布
"""
