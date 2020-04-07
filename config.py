#!/usr/bin/env python
# encoding: utf-8

"""
@version: python3.7
@author: 'sun'
@license: Apache Licence
@contact:
@software: PyCharm
@file: cinfig.py
@time: 2019/4/7 15:51
"""

import os
from argparse import Namespace
from collections import Counter
import json
import re
import string

import numpy as np
import pandas as pd
import torch
from utils import handle_dirs,set_seed_everywhere


args = Namespace(
    # Data and Path hyper parameters
    news_csv="/home/sun/文档/自然语言学习/数据集/ag_news_csv/news_with_splits.csv",
    vectorizer_file="vectorizer.json",
    model_state_file="model.pth",
    save_dir="checkpoints/model/cnn",
    # Model hyper parameters
    glove_filepath='/home/sun/文档/自然语言学习/数据集/glove.6B/glove.6B.100d.txt',
    use_glove=False,
    embedding_size=100,
    hidden_dim=100,
    num_channels=100,
    # Training hyper parameter
    seed=1337,
    learning_rate=0.001,
    dropout_p=0.5,
    batch_size=128,
    num_epochs=100,
    early_stopping_criteria=5,
    # Runtime option
    cuda=True,
    catch_keyboard_interrupt=True,
    reload_from_files=False,
    expand_filepaths_to_save_dir=True
)

if args.expand_filepaths_to_save_dir:
    args.vectorizer_file = os.path.join(args.save_dir,
                                        args.vectorizer_file)

    args.model_state_file = os.path.join(args.save_dir,
                                         args.model_state_file)

    print("Expanded filepaths: ")
    print("\t{}".format(args.vectorizer_file))
    print("\t{}".format(args.model_state_file))

# Check CUDA
if not torch.cuda.is_available():
    args.cuda = False

args.device = torch.device("cuda" if args.cuda else "cpu")
print("Using CUDA: {}".format(args.cuda))

# Set seed for reproducibility
set_seed_everywhere(args.seed, args.cuda)

# handle dirs
handle_dirs(args.save_dir)