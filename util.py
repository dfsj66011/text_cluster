# -*- coding:utf-8 -*-

"""
Create on 2020/9/4 2:22 下午
@Author: dfsj
@Description: 
"""
import re
from config import *


stop_words = set([item.strip() for item in open(PATH_STOPWORDS, 'r').readlines()])


def clear_character(sentence):
    """ 只保留汉字、字母、数字 """
    pattern = re.compile('[^\u4e00-\u9fa5^a-z^A-Z^0-9]')
    line = re.sub(pattern, '', sentence)
    new_sentence = ''.join(line.split())  # 去除空白
    return new_sentence


def drop_stopwords(line):
    """ 去停用词 """
    line_clean = []
    for word in line:
        if word in stop_words:
            continue
        line_clean.append(word)
    return line_clean

