# -*- coding:utf-8 -*-

"""
Create on 2020/10/16 10:50 上午
@Author: dfsj
@Description:   LDA 簇 预测
"""
from gensim.models.wrappers import LdaMallet
from gensim.corpora import Dictionary
from load_datas import Data


def lda_predict(texts):
    if isinstance(texts, str):
        texts = [texts]

    data = Data(bigram=False, trigram=False)                       # 预测阶段只分词
    texts = data.get_seg_corpus(texts)

    lda_model = LdaMallet.load("./data/lda.model")                 # 加载模型
    print(*lda_model.show_topics(num_topics=10, num_words=10, log=False, formatted=True), sep="\n")
    loaded_dct = Dictionary.load_from_text("./data/dictionary")    # 加载词典

    corpus = [loaded_dct.doc2bow(text) for text in texts]
    return lda_model[corpus]


if __name__ == "__main__":
    texts = """北京时间10月16日消息，据沃神报道，泰伦-卢将成为快船新任主教练，已与球队达成一份5年合同。快船方面认为，
    泰伦-卢的总冠军经历、在骑士季后赛的成功以及强大的沟通球员能力能够帮助快船弥补19-20赛季的一些缺憾。根据之前的报道，
    自里弗斯与快船分道扬镳以来，泰伦-卢一直在快船主教练的竞争中处于领先地位，并同时成为火箭和鹈鹕的主帅候选人。
    泰伦-卢曾担任凯尔特人、快船、骑士助教，在2015年-2018年担任骑士主教练，2015-16赛季率领骑士总决赛4-3击败勇士拿下队史首冠。
    此外据名记shames报道，昌西-比卢普斯将担任快船首席助理教练，前骑士主教练拉里-德鲁也加入担任助教。"""
    print(max(lda_predict(texts)[0], key=lambda k: k[1]))
