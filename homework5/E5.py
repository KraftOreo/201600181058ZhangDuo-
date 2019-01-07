from __future__ import print_function
import json
import sklearn.mixture
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn import cluster
import sklearn.cluster
from sklearn.metrics import v_measure_score, normalized_mutual_info_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

stoplist = stopwords.words('english')
PS = PorterStemmer()


# 读取twitter的json文件
def read_json(file_path):
    content = []
    cluster = []
    index = 0
    # 打开目标路径载入文件
    with open(file_path, 'r', errors='ignore') as f:
        for line in f:
            txt = json.loads(line)
            # 将每一条twitter的内容append到content里
            content.append(txt["text"])
            # 将每一条twitter的类append到cluster里
            cluster.append(txt["cluster"])
            index += 1
    # 返回content和cluster
    return content, cluster


# 处理文本内容
def process_twitter_text(content):
    text = content
    vectorizer = CountVectorizer()
    x = vectorizer.fit_transform(text)
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(x)
    # 返回向量化和tfidf化后的文本
    return tfidf.toarray()


# 用kmeans方法聚类
def k_means(tfidf, clusters):
    kmeans = cluster.KMeans(n_clusters=109).fit(tfidf)
    result = normalized_mutual_info_score(kmeans.labels_, clusters)
    # result = v_measure_score(kmeans.labels_, clusters)
    print("K-Means", result)


# 用affinity方法聚类
def affinity(tfidf, clusters):
    affinity = cluster.AffinityPropagation(preference=10).fit(tfidf)
    labels = affinity.labels_
    result = normalized_mutual_info_score(labels, clusters)
    # result = v_measure_score(labels, clusters)

    print("Affinity propagation", result)


# 用mean-shift方法聚类
def mean_shift(tfidf, clusters):
    meanshif = cluster.MeanShift(bandwidth=0.4, bin_seeding=2).fit(tfidf)
    result = normalized_mutual_info_score(meanshif.labels_, clusters)
    # result = v_measure_score(meanshif.labels_, clusters)

    print("Mean-shift", result)


# 用spectral方法聚类
def spectral(tfidf, clusters):
    spectra = cluster.SpectralClustering(109).fit(tfidf)
    result = normalized_mutual_info_score(spectra.labels_, clusters)
    # result = v_measure_score(spectra.labels_, clusters)

    print("Spectral", result)


# 用ward方法聚类
def ward(tfidf, clusters):
    ward = cluster.AgglomerativeClustering(109, linkage='ward').fit((tfidf))
    result = normalized_mutual_info_score(ward.labels_, clusters)
    # result = v_measure_score(ward.labels_, clusters)
    print("Ward", result)


# 用agglomerative方法聚类
def agg(tfidf, clusters):
    agg = cluster.AgglomerativeClustering(n_clusters=109, linkage='average').fit_predict(tfidf)
    result = normalized_mutual_info_score(agg, clusters)
    # result = v_measure_score(agg, clusters)

    print("Aggiomerative", result)


# 用DBSCAN方法聚类
def dbscan(tfidf, clusters):
    scan = cluster.DBSCAN(eps=0.5, min_samples=1).fit(tfidf)
    result = normalized_mutual_info_score(scan.labels_, clusters)
    # result = v_measure_score(scan.labels_, clusters)

    print("DBSCAN", result)


# 用高斯混合模型聚类
def gaussian_mixtures(tfidf, clusters):
    gaussian = sklearn.mixture.GaussianMixture(n_components=10).fit(tfidf)
    labels = gaussian.predict(tfidf)
    result = normalized_mutual_info_score(labels, clusters)
    # result = v_measure_score(labels, clusters)

    print("Gaussian mixtures", result)


if __name__ == "__main__":
    file_path = "Homework5Tweets.txt"
    content, clusters = read_json(file_path)
    tfidf = process_twitter_text(content)
    k_means(tfidf, clusters)
    affinity(tfidf, clusters)
    mean_shift(tfidf, clusters)
    spectral(tfidf, clusters)
    ward(tfidf, clusters)
    agg(tfidf, clusters)
    dbscan(tfidf, clusters)
    gaussian_mixtures(tfidf, clusters)
