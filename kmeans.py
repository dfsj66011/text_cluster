# -*- coding:utf-8 -*-

"""
Create on 2020/9/28 4:13 下午
@Author: dfsj
@Description:  Kmeans 文本聚类
"""
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer, TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn import metrics
import matplotlib.pyplot as plt

from config import *


class KMEANS:
    """ KMeans 文本聚类算法 """

    vectorizer = None
    X = None
    km = None
    svd = None

    def __init__(self, texts, num_clusters=10, minibatch=True, n_components=100,
                 n_features=250000, use_hashing=False, use_idf=True):
        """
        :param texts:           聚类文本
        :param num_clusters:    聚类数
        :param minibatch:       是否是否 MiniBatchKMeans
        :param n_components:    使用潜在语义分析处理文档，可以设置为 None 不进行压缩
        :param n_features:      特征（维度）的最大数量，特征压缩，只用于 hash 特征表示
        :param use_hashing:     hash 特征向量
        :param use_idf:         是否使用逆文档频率特征
        """
        self.texts = texts
        self.num_clusters = num_clusters
        self.minibatch = minibatch
        self.n_components = n_components
        self.n_features = n_features
        self.use_hashing = use_hashing
        self.use_idf = use_idf
        self.text2vec()

    def text2vec(self):
        """ 文本向量化表示 """
        if self.use_hashing:
            if self.use_idf:
                # Perform an IDF normalization on the output of HashingVectorizer
                hasher = HashingVectorizer(n_features=self.n_features, alternate_sign=False, norm=None)
                self.vectorizer = make_pipeline(hasher, TfidfTransformer())
            else:
                self.vectorizer = HashingVectorizer(n_features=self.n_features, alternate_sign=False, norm='l2')
        else:
            self.vectorizer = TfidfVectorizer(max_df=0.5, min_df=2,  use_idf=self.use_idf)
        self.X = self.vectorizer.fit_transform(self.texts)
        logger.info("n_samples: %d, n_features: %d" % self.X.shape)

        if self.n_components:
            logger.info("Performing dimensionality reduction using LSA")
            # Vectorizer results are normalized, which makes KMeans behave as
            # spherical k-means for better results. Since LSA/SVD results are
            # not normalized, we have to redo the normalization.
            self.svd = TruncatedSVD(self.n_components)
            normalizer = Normalizer(copy=False)
            lsa = make_pipeline(self.svd, normalizer)
            self.X = lsa.fit_transform(self.X)
            explained_variance = self.svd.explained_variance_ratio_.sum()
            logger.info("Explained variance of the SVD step: {}%".format(int(explained_variance * 100)))

    def train(self):
        if self.minibatch:
            self.km = MiniBatchKMeans(n_clusters=self.num_clusters, init='k-means++', n_init=1,
                                      init_size=1000, batch_size=1000, verbose=False)
        else:
            self.km = KMeans(n_clusters=self.num_clusters, init='k-means++', max_iter=100, n_init=1, verbose=False)

        self.km.fit(self.X)
        return self.km

    def print_top_terms(self, top_n=10):
        if not self.use_hashing:
            if not self.km:
                _ = self.train()
            logger.info("Top terms per cluster:")
            if self.n_components:
                original_space_centroids = self.svd.inverse_transform(self.km.cluster_centers_)
                order_centroids = original_space_centroids.argsort()[:, ::-1]
            else:
                order_centroids = self.km.cluster_centers_.argsort()[:, ::-1]

            terms = self.vectorizer.get_feature_names()
            for i in range(self.num_clusters):
                res = []
                for ind in order_centroids[i, :top_n]:
                    res.append(terms[ind])
                logger.info("Cluster {}: {}".format(i, " ".join(res)))
        else:
            logger.warning("hash 编码方式不支持该方法")

    def print_summary(self, labels=None):
        """ labels 为该数据集的真实类别标签，真实数据可能不存在该标签，因此部分指标可能不可用 """
        if not self.km:
            _ = self.train()
        if labels is not None:
            logger.info("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, self.km.labels_))
            logger.info("Completeness: %0.3f" % metrics.completeness_score(labels, self.km.labels_))
            logger.info("V-measure: %0.3f" % metrics.v_measure_score(labels, self.km.labels_))
            logger.info("Adjusted Rand-Index: %.3f" % metrics.adjusted_rand_score(labels, self.km.labels_))
        logger.info("Silhouette Coefficient: %0.3f" %
                    metrics.silhouette_score(self.X, self.km.labels_, metric='euclidean'))

        result = list(self.km.predict(self.X))
        logger.info('Cluster distribution:')
        logger.info(dict([(i, result.count(i)) for i in result]))
        logger.info(-self.km.score(self.X))

    def find_optimal_clusters(self, max_k):
        iters = range(2, max_k + 1, 2)

        sse = []
        for k in iters:
            sse.append(
                MiniBatchKMeans(n_clusters=k, init="k-means++", init_size=1024, batch_size=2048, random_state=20).fit(
                    self.X).inertia_)
            logger.info('Fit {} clusters'.format(k))

        f, ax = plt.subplots(1, 1)
        ax.plot(iters, sse, marker='o')
        ax.set_xlabel('Cluster Centers')
        ax.set_xticks(iters)
        ax.set_xticklabels(iters)
        ax.set_ylabel('SSE')
        ax.set_title('SSE by Cluster Center Plot')
        plt.savefig(PIC_KMEANS)
