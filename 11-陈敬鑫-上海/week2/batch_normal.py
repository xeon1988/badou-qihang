#!/usr/bin/python3
# -*- coding:utf-8 -*-
# @time     : 2021/4/28 17:19
# @Author   : ReidChen
# Document  ：Batch Normalization，对神经网络每一层输入进行归一化处理，可避免梯度消失

import torch
import numpy as np

# torch实现
x = torch.randn(4,8)
bn = torch.nn.BatchNorm1d(8)
y = bn(x)
print(y, "torch bn输出")

# 取torch bn weight 为激活参数
w = bn.state_dict()["weight"]
b = bn.state_dict()["bias"]

# 计算x 的均值与方差
esp = 1e-5
x_mean = torch.mean(x,dim=0)
x_std = torch.mean(torch.square(x - x_mean), dim=0)
x = (x-x_mean) / torch.sqrt(x_std + esp)
y = w * x + b
print(y, "自定义输出")

