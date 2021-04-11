"""
离散分布采样
例如令p(x)=[0.1,0.2,0.3,0.4]，
那么我们可以把概率分布的向量看做一个区间段，然后从x ∼ U ( 0 , 1 )分布中采样，判断x落在哪个区间内。
区间长度与概率成正比，这样当采样的次数越多，采样就越符合原来的分布。
举个例子，假设p(x)=[0.1,0.2,0.3,0.4]，
分别为"hello", “java”, “python”, "scala"四个词出现的概率。我们按照上面的算法实现看看最终的结果

原文链接：https://blog.csdn.net/bitcarmanlee/article/details/82795137
"""
import numpy as np
from collections import defaultdict

dic = defaultdict(int)


def sample():
    u = np.random.rand()
    # 通过本函数可以返回一个或一组服从“0~1”均匀分布的随机样本值。随机样本取值范围是[0, 1)，不包括1。
    if u <= 0.1:
        dic["hello"] += 1
    elif u <= 0.3:
        dic["java"] += 1
    elif u <= 0.6:
        dic["python"] += 1
    else:
        dic["scala"] += 1


for i in range(10000):
    sample()
for k, v in dic.items():
    print(k, v)

"""
hello 1032
java 1990
python 2977
scala 4001

通过结果可以知道，抽样生成的样本：符合[0.1,0.2,0.3,0.4]分布
"""
