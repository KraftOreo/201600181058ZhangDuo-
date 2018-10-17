import os
import math
import chardet
from gensim import corpora
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import random


# derive the encoding format of the data
def get_encoding(file):
    # 二进制方式读取，获取字节数据，检测类型
    with open(file, 'rb') as f:
        return chardet.detect(f.read())['encoding']


# build every single species dataset
def build_classification_dataset(path):
    contents = []
    files = os.listdir(path)
    random.shuffle(files)
    ratio = math.floor(0.7 * len(files))
    file_train = files[:ratio]
    file_test = files[ratio:]
    for file in file_train:
        with open(path + '\\' + file, 'rb') as text:
            all_contents = text.readlines()
            code_format = get_encoding(path + '\\' + file)
            if not code_format == None:
                for content in all_contents:
                    decoded_content = content.decode(code_format)
                    decoded_content = str(decoded_content)
                    contents.append(decoded_content)
    contents = ' '.join(contents)
    test_path = []
    for file in file_test:
        test_path.append(path + '\\' + file)
    return contents, test_path


# dataset of every file
def build_small_dataset(path):
    contents = []
    with open(path, 'rb') as text:
        all_contents = text.readlines()
        code_format = get_encoding(path)
        if not code_format == None:
            for content in all_contents:
                decoded_content = content.decode(code_format)
                decoded_content = str(decoded_content)
                contents.append(decoded_content)
        contents = ' '.join(contents)
    return contents


# to derive a better and cleaner dataset
def data_processing(dataset):
    dataset_lower = [data.lower() for data in dataset]
    dataset_tokenized = [word_tokenize(data) for data in dataset_lower]
    stop_words = stopwords.words('english')
    dataset_filtered = [words for words in dataset_tokenized if words not in stop_words]
    english_styled_punctuations = [',', '.''?', '/', ':', '<', '>', '[', ']', '(', ')', '*', '&', '^', '%', '$', '#',
                                   '@', '!', '~', '`', '+', '_', '-']
    dataset_with_no_punctuations = [[words for words in data if words not in english_styled_punctuations] for data in
                                    dataset_filtered]
    return dataset_with_no_punctuations


def data_stemming(dataset):
    ps = PorterStemmer()
    dataset_stemmed = [[ps.stem(word) for word in data] for data in dataset]
    return dataset_stemmed


def build_corpus(dataset):
    dataset_dictionary = corpora.Dictionary(dataset)
    corpus = [dataset_dictionary.doc2bow(word) for word in dataset]
    return corpus, dataset_dictionary


def build_vs_model(dataset):
    data_processed = data_processing(dataset)
    dataset_stemmed = data_stemming(data_processed)
    corpus, dictionary = build_corpus(dataset_stemmed)
    return dictionary, corpus


if __name__ == '__main__':
    path = "C:\\Users\\97653\\Desktop\\人工智能\\IR&DM\\20news-19997\\20_newsgroups"  # 文件路径
    dataset_overall = []  # 创建空列表来存放数据集
    path_to_test = {}  # 创建空字典，存放每一类的测试集的文件路径
    for dir in os.listdir(path):
        files = os.listdir(path + '\\' + dir)
        dataset, test_path = build_classification_dataset(path + '\\' + dir)
        dataset_overall.append(dataset)  # 创建总体数据集，长度为20，每一个元素都是这一类的处理后的数据
        path_to_test[dir] = test_path  # 添加路径到字典
    dictionary, corpus = build_vs_model(dataset_overall)  # 构造语料库和字典
    true_r = 0  # 正确结果的数量
    false_r = 0  # 错误结果的数量
    for dirs in path_to_test.keys():
        i = list(path_to_test.keys()).index(dirs)  # 类的编号
        for test_sample in path_to_test[dirs]:  # 取出测试数据并打造测试数据的小语料库
            sample_data = build_small_dataset(test_sample)
            sample_dictionary, sample_corpus = build_vs_model(sample_data)
            for key in sample_dictionary.token2id.keys():  # 如果测试集的词在大的语料库里
                if key in dictionary.token2id.keys():
                    idx = dictionary.token2id[key]  # 将这个词在大字典中的位置标记出来
                    prob_list = []
                    for corpus_class in corpus:  # 对每一个类的小语料库进行操作
                        SUM = 0  # 所有词出现的次数
                        SUM_imagine = len(corpus_class)  # 平滑用的补充的先验
                        p = 0  # 起初概率为1，对数值为0
                        for tuples in corpus_class:
                            SUM += tuples[1]  # 平滑补充的数量
                        tuples = [tuples[0] for tuples in corpus_class]  # 词的索引值的列表
                        if idx in tuples:  # 如果这个词在列表中
                            for tuples in corpus_class:  # 计算概率，使用平滑
                                p += math.log((tuples[1] + 1) / (SUM + SUM_imagine))
                        else:
                            p += math.log(1 / (SUM + SUM_imagine))
                        prob_list.append(p)  # 将测试集属于这二十类的概率保存下来
                    a = prob_list.index(max(prob_list))  # 最大概率的类
                    if i == a:  # 如果类的编号等于最大概率的类的编号
                        true_r += 1  # 则分类正确，正确数加1
                    else:  # 如果不一致
                        false_r += 1  # 错误数加1
    print(true_r / (true_r + false_r))  # 输出正确率
    # 82%
