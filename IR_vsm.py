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
    path = "C:\\Users\\97653\\Desktop\\人工智能\\IR&DM\\20news-19997\\20_newsgroups"
    dict_list = []
    dataset_overall = []
    path_to_test = {}
    for dir in os.listdir(path):
        files = os.listdir(path + '\\' + dir)
        dataset, test_path = build_classification_dataset(path + '\\' + dir)
        dataset_overall.append(dataset)
        path_to_test[dir] = test_path
    dictionary, corpus = build_vs_model(dataset_overall)

    prob = math.log(1 / 20)
    true = 0
    false = 0
    for dirs in path_to_test.keys():
        i = list(path_to_test.keys()).index(dirs)
        print(i)
        for test_sample in path_to_test[dirs]:
            sample_data = build_small_dataset(test_sample)
            sample_dictionary, sample_corpus = build_vs_model(sample_data)
            for key in sample_dictionary.token2id.keys():
                if key in dictionary.token2id.keys():
                    idx = dictionary.token2id[key]
                    prob_list = []
                    for corpus_class in corpus:
                        SUM = 0
                        SUM_imagine = len(corpus_class)
                        p = 0
                        for tuples in corpus_class:
                            SUM += tuples[1]
                        for tuples in corpus_class:
                            if tuples[0] == idx:
                                p += math.log(tuples[1] + 1 / SUM + SUM_imagine)
                            else:
                                p += math.log(1 / SUM + SUM_imagine)
                        prob_list.append(p)
                    print(prob_list.index(max(prob_list)))
                    a = prob_list.index(max(prob_list))
                    if i == a:
                        true += 1
                    else:
                        false += 1
    print(true / (true + false))
