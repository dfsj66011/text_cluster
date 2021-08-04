# -*- coding:utf-8 -*-

"""
Create on 2020/9/27 9:15 上午
@Author: dfsj
@Description:   LDA 模型训练
"""
from load_datas import Data
from lda import LDA
from kmeans import KMEANS
import joblib
from config import *


data = Data(TRAIN_SETS)
texts = data.get_seg_corpus()
labels = data.get_labels()
# with open("texts", "w") as f:
#     f.write("\n".join([" ".join(items) for items in texts]))
# texts = [item.strip().split() for item in open("data/texts", "r").readlines()]


def train_lda():
    lda = LDA(texts=texts, num_topics=10)
    model = lda.train()
    model.save(MODEL_LDA)


def train_kmeans():
    corpus = [' '.join(line) for line in texts]
    kmeans = KMEANS(corpus)
    model = kmeans.train()
    joblib.dump(model, MODEL_KMEANS)

    kmeans.find_optimal_clusters(20)
    kmeans.print_top_terms()
    kmeans.print_summary(labels=labels)


if __name__ == "__main__":
    train_lda()
    # train_kmeans()
