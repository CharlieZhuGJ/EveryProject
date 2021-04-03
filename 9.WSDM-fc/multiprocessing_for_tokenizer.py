import pandas as pd
import numpy as np
import time
import jieba.posseg as pseg
from multiprocessing import cpu_count, Pool


def jieba_tokenizer(text):
    if not isinstance(text, str):
        text = str(text)
    words = pseg.cut(text)
    return ' '.join([
        word for word, flag in words if flag != 'x'])


def process(data):
    res = data.apply(jieba_tokenizer)
    return res


def check_merge_idx(data, res):
    assert (data.index == res.index).all()


def parallelize(data, func):
    cores = partitions = cpu_count()
    data_split = np.array_split(data, partitions)
    pool = Pool(cores)
    res = pd.concat(pool.map(func, data_split))
    pool.close()
    pool.join()
    check_merge_idx(data, res)
    return res


if __name__ == '__main__':
    train = pd.read_csv("test.csv", index_col='id')
    print(train.head(3))

    print("start to training")
    start = time.time()
    train['title1_tokenized'] = parallelize(train.loc[:, 'title1_zh'], process)
    mid = time.time()
    print(mid - start)
    # train.to_csv('tokenized_train.csv', index=True)
    train['title2_tokenized'] = parallelize(train.loc[:, 'title2_zh'], process)
    end = time.time()
    print(end - mid)
    train.to_csv('tokenized_test.csv', index=True)

# # 一个子进程

#
# # 多个子进程
import multiprocessing
import time

def func(msg):
    print(multiprocessing.current_process().name + '-' + msg)

if __name__ == "__main__":
    pool = multiprocessing.Pool(processes=4) # 创建4个进程
    for i in range(10):
        msg = "hello %d" %(i)
        pool.apply_async(func, (msg, ))
    pool.close() # 关闭进程池，表示不能在往进程池中添加进程
    pool.join() # 等待进程池中的所有进程执行完毕，必须在close()之后调用
    print("Sub-process(es) done.")
from multiprocessing import Process

def f(name):
    print('hello', name)

if __name__ == '__main__':
    p = Process(target=f, args=('bob',)) # 新建一个子进程p，目标函数是f，args是函数f的参数列表
    p.start() # 开始执行进程
    p.join() # 等待子进程结束
# 多个子进程并返回值
import multiprocessing
import time


def func(msg):
    return multiprocessing.current_process().name + '-' + msg


if __name__ == "__main__":
    pool = multiprocessing.Pool(processes=4)  # 创建4个进程
    results = []
    for i in range(10):
        msg = "hello %d" % (i)
        # results.append(pool.apply_async(func, (msg,)))
        results.append(pool.map(func, (msg,)))
    pool.close()  # 关闭进程池，表示不能再往进程池中添加进程，需要在join之前调用
    pool.join()  # 等待进程池中的所有进程执行完毕
    print("Sub-process(es) done.")

    for res in results:
        # print(res.get())
        print(res)

from multiprocessing import Pool, Manager


def func(que, i):
    print('sub process running:' + str(i))
    que.put('sub process info:' + str(i))


if __name__ == '__main__':
    process_pool = Pool(4)
    que = Manager().Queue()
    for i in range(4):
        process_pool.apply_async(func, args=(que, i))
    process_pool.close()
    process_pool.join()
    print('sub processes finished')
    for i in range(4):
        print(que.get())
