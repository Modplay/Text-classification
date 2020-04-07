#!/usr/bin/env python
# encoding: utf-8

"""
@version: python3.7
@author: 'sun'
@license: Apache Licence
@contact:
@software: PyCharm
@file: visualize.py
@time: 2019/4/7 15:51
"""


import time
import matplotlib.pyplot as plt




def plot_performance(self,train_state):
    # 设置图大小
    plt.figure(figsize=(15, 5))

    # 画出损失
    plt.subplot(1, 2, 1)
    plt.title("Loss")
    plt.plot(train_state['train_loss'], label="train")
    plt.plot(train_state['val_loss'], label="val")
    plt.legend(loc='upper right')

    # 画出准确率
    plt.subplot(1, 2, 2)
    plt.title("Accuracy")
    plt.plot(train_state['train_acc'], label="train")
    plt.plot(train_state['val_acc'], label="val")
    plt.legend(loc='lower right')

    # 存图
    # plt.savefig(os.path.join(self.save_dir, "performance.png"))

    # 展示图
    plt.show()
