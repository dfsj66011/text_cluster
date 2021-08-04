# -*- coding:utf-8 -*-

"""
Create on 2020/9/4 11:08 上午
@Author: dfsj
@Description:  LDA 主题模型
"""
import pandas as pd
import gensim
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel
import matplotlib.pyplot as plt
from gensim.models.wrappers import LdaMallet

from config import *


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('max_colwidth', 80)


def draw_train_result(x, coherence_values):
    # Print the coherence scores
    for m, cv in zip(x, coherence_values):
        logger.info("Num Topics =", m, " has Coherence Value of", round(cv, 4))

    plt.plot(x, coherence_values)
    plt.xlabel("Num Topics")
    plt.ylabel("Coherence score")
    plt.legend("coherence_values", loc='best')
    plt.savefig(PIC_LDA)


def get_best_model_index(x, coherence_values: list):
    """
    选择最佳模型，目前下方自动化选择最佳模型的想法还不够成熟，此处目前最好人工选择模型
    实际上最好的不一定是 coherence_value 得分最高的，而是较为平滑曲线的左侧端点，
    此处为了简单，选择具有最高 coherence_value 的模型
    """
    for m, cv in zip(x, coherence_values):
        logger.info("Num Topics =", m, " has Coherence Value of", round(cv, 4))
    max_value = max(coherence_values)
    max_value_index = coherence_values.index(max_value)

    # index = max_value_index
    # while index >= 0 and abs(coherence_values[index - 1] - coherence_values[index]) < 0.002:
    #     index -= 1
    # return index
    return max_value_index


class LDA:

    best_model: LdaMallet
    id2word: Dictionary
    corpus: list

    def __init__(self, texts, num_topics=None, start=5, limit=15, step=5, print_summary=True, workers=1):
        """
        LDA 模型
        :param texts:         经过分词后的文本表示形式
        :param num_topics:    选定主题数，如果提供该值，则不进行超参搜索
        :param start:         当不提供主题数，根据指定的超参范围进行搜索，起始
        :param limit:         终止
        :param step:          步长
        """
        self.num_topics = num_topics
        self.start = start
        self.limit = limit
        self.step = step
        self.texts = texts
        self.print_summary = print_summary
        self.workers = workers
        self.data_process()

    def data_process(self):
        logger.info("数据转化中，请稍后 ...")
        self.id2word = Dictionary(self.texts)                              # 生成字典
        self.id2word.save_as_text(DICT_NAME)                               # 保存字典
        logger.info("字典生成完毕，示例：id2word[0] = {}".format(self.id2word[0]))
        self.corpus = [self.id2word.doc2bow(text) for text in self.texts]  # 生成 LDA 所需语料形式
        logger.info("数据转化完成，示例：corpus[0] = {}".format(self.corpus[0]))
        logger.info("人为可读转化，corpus[0] = {}".format([(self.id2word[index], freq) for index, freq in self.corpus[0]]))

    def compute_coherence_values(self, limit, start=2, step=3):
        """
        Compute c_v coherence for various number of topics

        Parameters:
        ----------
        dictionary : Gensim dictionary
        corpus : Gensim corpus
        texts : List of input texts
        limit : Max num of topics

        Returns:
        -------
        model_list : List of LDA topic models
        coherence_values : Coherence values corresponding to the LDA model with respective number of topics
        """
        coherence_values = []
        model_list = []
        for num_topics in range(start, limit, step):
            model = gensim.models.wrappers.LdaMallet(
                PATH_MALLET, corpus=self.corpus, num_topics=num_topics, id2word=self.id2word, workers=self.workers)
            model_list.append(model)
            coherencemodel = CoherenceModel(model=model, texts=self.texts, dictionary=self.id2word, coherence='c_v')
            coherence_values.append(coherencemodel.get_coherence())

        return model_list, coherence_values

    def train(self):
        if self.num_topics:
            self.best_model = gensim.models.wrappers.LdaMallet(
                PATH_MALLET, corpus=self.corpus, num_topics=self.num_topics,
                id2word=self.id2word, workers=self.workers, prefix=PREFIX)

            coherencemodel = CoherenceModel(
                model=self.best_model, texts=self.texts, dictionary=self.id2word, coherence='c_v')

            coherence_ldamallet = coherencemodel.get_coherence()
            logger.info('Coherence Score: {}'.format(coherence_ldamallet))
        else:
            model_list, coherence_values = self.compute_coherence_values(start=5, limit=15, step=5)
            x = range(self.start, self.limit, self.step)
            draw_train_result(x, coherence_values)                                              # 画图

            # ****************************************************************************************
            self.best_model = model_list[get_best_model_index(x, coherence_values)]
            # 目前自动化选择最佳模型的想法还不成熟
            # ****************************************************************************************

        self.print_topic()
        if self.print_summary:
            self.get_summary()
        logger.info(self.best_model[self.corpus[0]])
        return self.best_model

    def print_topic(self):
        # model_topics = self.best_model.show_topics(formatted=False)
        logger.info(self.best_model.print_topics(num_words=10))

    def format_topics_sentences(self):
        """ 对每一条数据进行主题标注，并给出该主题的百分比以及关键词内容等 """
        sent_topics_df = pd.DataFrame()

        # Get main topic in each document
        for i, row in enumerate(self.best_model[self.corpus]):
            row = sorted(row, key=lambda x: (x[1]), reverse=True)
            # Get the Dominant topic, Perc Contribution and Keywords for each document
            for j, (topic_num, prop_topic) in enumerate(row):
                if j == 0:  # => dominant topic
                    wp = self.best_model.show_topic(topic_num)
                    topic_keywords = ", ".join([word for word, prop in wp])
                    sent_topics_df = sent_topics_df.append(
                        pd.Series([int(topic_num), round(prop_topic, 4), topic_keywords]), ignore_index=True)
                else:
                    break
        sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']
        contents = pd.Series(self.texts)
        sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
        return sent_topics_df

    def get_summary(self):
        """ 打印统计信息 """
        sent_topics_sorteddf_mallet = pd.DataFrame()

        df_topic_sents_keywords = self.format_topics_sentences()                    # 得到每一条数据的主题信息
        sent_topics_outdf_grpd = df_topic_sents_keywords.groupby('Dominant_Topic')  # 以主题分组

        for _, grp in sent_topics_outdf_grpd:
            sent_topics_sorteddf_mallet = pd.concat(
                [sent_topics_sorteddf_mallet,
                 grp.sort_values(['Perc_Contribution'], ascending=[0]).head(1)], axis=0)   # 只选择贡献度最高的一个
        sent_topics_sorteddf_mallet.reset_index(drop=True, inplace=True)
        sent_topics_sorteddf_mallet.columns = ['Topic_Num', "Topic_Perc_Contrib", "Keywords", "Text"]

        topic_counts = df_topic_sents_keywords['Dominant_Topic'].value_counts()     # 每个主题的文档数
        topic_contribution = round(topic_counts / topic_counts.sum(), 4)            # 每个主题的文档占比

        topic_num_keywords = df_topic_sents_keywords[
            ['Dominant_Topic', 'Topic_Keywords']].drop_duplicates().reset_index(drop=True)

        df_dominant_topics = pd.concat([topic_num_keywords, topic_counts, topic_contribution], axis=1)
        df_dominant_topics.columns = ['Topic_Num', 'Keywords', 'Num_Documents', 'Perc_Documents']
        df_dominant_topics.sort_values(by="Topic_Num", ascending=True, inplace=True)
        summary = pd.merge(sent_topics_sorteddf_mallet, df_dominant_topics, on=["Topic_Num", "Keywords"])
        logger.info(summary)
