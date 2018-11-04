import os
import math
import chardet
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
    english_styled_punctuations = [',', '.', '?', '/', ':', '<', '>', '[', ']', '(', ')', '*', '&', '^', '%', '$', '#',
                                   '@', '!', '~', '`', '+', '_', '-', '“', '”', '~~', '``', '`', '{', '}', '"',
                                   '\\', '…', '\'', '\"', '\'\'', '...', '--', '~~~', 'o', '|', ';', 'the', 'will',
                                   'be', 'in', 'is', 'and', 'to', 'are', 'not', 'my', 'you', 'he', 'she', 'they', '\'s',
                                   'of', 'form', 'would', 'do', 'can', 'could', 'may', 'might', 'think', 'must', 'have',
                                   'did', 'done', 'does', 'don\'t', 'by', 'with', 'they', 'on', 'a', 'or', 'some', 'no',
                                   'i', 'does\'t', 'did\'t', 'can\'t', 'couldn\'t', 'wouldn\'t', 't', '\'t', 'am',
                                   '\'m', 'm', 'this', 'that', 'these', 'those', 'as', 'also', '=', 'we', 'within',
                                   'without', 'their', 'here', 'there', 're', '\'re', 'his', 'her', 'your', 'll',
                                   '\'ll', 'xref', 'so', 'very', 'much', 'for', 'when', 'while', 'which', 'how',
                                   'n\'t''from', 'than', '*******sig', 'them', 'if', 've', '\'ve', 'v', 'been', "'ve",
                                   'it', 'what', 'but', 'an', 'wa', 'was', 'were', 'who', 'just', "n't", 'thi', 'at']
    dataset_with_no_punctuations = [[words for words in data if words not in english_styled_punctuations] for data in
                                    dataset_filtered]
    ps = PorterStemmer()
    dataset_stemmed = [[ps.stem(word) for word in data] for data in dataset_with_no_punctuations]
    return dataset_stemmed


if __name__ == '__main__':
    '''
    构造字典
    '''
    path = "C:\\Users\\97653\\Desktop\\人工智能\\IR&DM\\20news-19997\\20_newsgroups"  # 文件路径
    dataset_overall = []  # 创建空列表来存放数据集
    path_to_test = {}  # 创建空字典，存放每一类的测试集的文件路径
    for dir in os.listdir(path):
        dict = {"NO": os.listdir(path).index(dir)}
        files = os.listdir(path + '\\' + dir)
        dataset, test_path = build_classification_dataset(path + '\\' + dir)
        dict['dataset'] = [dataset]
        dataset_overall.append(dict)  # 创建总体数据集，长度为20，每一个元素都是这一类的处理后的数据
        path_to_test[os.listdir(path).index(dir)] = test_path  # 添加路径到字典
    for i in dataset_overall:
        i['dataset'] = data_processing(i['dataset'])
        print(len(i['dataset'][0]))
        dict = {}
        for j in i['dataset'][0]:
            if j not in dict.keys():
                dict[j] = 1
            else:
                dict[j] += 1
        i['dataset'] = dict

    '''
    测试阶段
    '''
    true = 0
    false = 0
    time = 0
    for key in path_to_test.keys():
        for path in path_to_test[key]:
            p = 0
            dict = {}
            dataset = [build_small_dataset(path)]
            dataset = data_processing(dataset)
            for j in dataset[0]:
                if j not in dict.keys():
                    dict[j] = 1
                else:
                    dict[j] += 1
            prob = []
            for i in dataset_overall:
                for k in i['dataset'].keys():
                    if k in dict.keys():
                        p += dict[k] * math.log(
                            (i['dataset'][k] + 1) / (sum([*i['dataset'].values()]) + len([*i['dataset'].keys()])))
                    else:
                        p += dict[k] * math.log(1 / (sum([*i['dataset'].values()]) + len([*i['dataset'].keys()])))

            prob.append(p)
            print('one file done')
            print(prob.index(max(prob)))
            print([*path_to_test.keys()].index(key))
            if prob.index(max(prob)) == [*path_to_test.keys()].index(key):
                true += 1
            else:
                false += 1
            time += 1
            print(time)
            print(true / (true + false))
