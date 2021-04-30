#!/usr/bin/python3
# -*- coding:utf-8 -*-
# @time     : 2021/4/28 16:18
# @Author   : ReidChen
# Document  ：余弦相似度计算


import numpy as np

# 老师的方法
def cosine_distance(x, y):
    return np.sum(x * y) / np.sqrt(np.sum(np.square(x)) * np.sum(np.square(y)))

# linalg.norm ,ord=2 二范式
def cos_sim(a, b):
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    return np.dot(a,b)/(a_norm * b_norm)

# 常规向量计算余弦相似度
x = np.array([2,5,8])
y = np.array([3,6,9])
print(cosine_distance(x,y))
print(cos_sim(x,y))
print('-----------我是分割线---------')
# 同方向向量计算
x = np.array([2,5,8])
y = np.array([4,10,16])
print(cosine_distance(x,y))
print(cos_sim(x,y))