# -*- coding:utf-8 -*-

"""
Create on 2020/9/11 2:20 下午
@Author: dfsj
@Description: 
"""
import os
import logging as logger
from gensim.test.utils import get_tmpfile


BASE_PATH = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_PATH, "data")

PATH_STOPWORDS = os.path.join(DATA_PATH, "stopwords.txt")
PATH_MALLET = os.path.join(DATA_PATH, 'mallet-2.0.8', "bin", "mallet")  # Download File: http://mallet.cs.umass.edu/dist/mallet-2.0.8.zip

# 日志
LOG_FILE = os.path.join(BASE_PATH, "log", "server.log")
LOG_FORMAT = "%(asctime)s - %(threadName)s - %(name)s - %(levelname)s - %(message)s"    # 日志格式化输出
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"                        # 日期格式
fp = logger.FileHandler(LOG_FILE, encoding='utf-8')
fs = logger.StreamHandler()
logger.basicConfig(level=logger.INFO, format=LOG_FORMAT, datefmt=DATE_FORMAT, handlers=[fp, fs])

# PIC && MODEL SAVE
PIC_LDA = os.path.join(DATA_PATH, "lda_train_result.png")
PIC_KMEANS = os.path.join(DATA_PATH, "kmeans_train_result.png")
MODEL_LDA = os.path.join(DATA_PATH, "lda.model")
MODEL_KMEANS = os.path.join(DATA_PATH, "kmeans.model")


# DataSet

TRAIN_SETS = "/xxxxxxx/datasets/cnews/cnews.train.txt"   # 数据集存储位置

# lda 词典
DICT_NAME = get_tmpfile(os.path.join(DATA_PATH, "dictionary"))
PREFIX = os.path.join(DATA_PATH, "temporary", "temporary_")
