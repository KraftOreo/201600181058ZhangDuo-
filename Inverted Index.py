from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import json


def data_processing(dataset):
    dataset_lower = [data.lower() for data in dataset]
    dataset_tokenized = [word_tokenize(data) for data in dataset_lower]
    stop_words = stopwords.words('english')
    dataset_filtered = [words for words in dataset_tokenized if words not in stop_words]
    english_styled_punctuations = [',', '.''?', '/', ':', '<', '>', '[', ']', '(', ')', '*', '&', '^', '%', '$', '#',
                                   '@', '!', '~', '`', '+', '_', '-', '“', '”', '~~', '``', '`', '{', '}', str(''), '"',
                                   '\\', '…', '\'', '\"', '\'\'', '...']
    dataset_with_no_punctuations = [[words for words in data if words not in english_styled_punctuations] for data in
                                    dataset_filtered]
    return dataset_with_no_punctuations


def data_stemming(dataset):
    ps = PorterStemmer()
    dataset_stemmed = [ps.stem(word) for word in dataset]
    return dataset_stemmed


def dict_construction(path):
    contents = []
    with open(path) as f:
        content = f.readlines()
        contents.append(content)
    tweets = []
    a = contents[0]
    for i in range(len(a)):
        dict = {'tweet NO': i}
        dataset = [a[i]]
        dataset = data_processing(dataset)
        dataset = dataset[0]
        data = dataset[dataset.index('username') + 1:dataset.index('clusterno')]
        if 'http' not in dataset:
            text = dataset[dataset.index('text') + 1:dataset.index('timestr')]
        else:
            del dataset[dataset.index('http'):dataset.index('timestr')]
            text = dataset[dataset.index('text') + 1:dataset.index('timestr')]
        text = data_stemming(text)
        data.extend(text)
        time = dataset[dataset.index('timestr') + 1:dataset.index('+0000')]
        data.extend(time)
        dict['data'] = data
        tweets.append(dict)
    return tweets


def build_inverted_index(tweets):
    dict_inv = {}
    for i in tweets:
        for j in i['data']:
            if j not in dict_inv.keys():
                dict_inv[j] = [i['tweet NO']]
            else:
                dict_inv[j].append(i['tweet NO'])
    return dict_inv


def merge(dict_inv, query):
    query = query_process(query)
    merge_list = []
    for i in query:
        merge_list.append(dict_inv[i])
    while len(merge_list) != 1:
        merge_list_small = []
        for i in range(1, len(merge_list)):
            temp = []
            for item1 in merge_list[0]:
                for item2 in merge_list[i]:
                    if item1 == item2:
                        temp.append(item1)
                        break
            merge_list_small.append(temp)
        merge_list = merge_list_small
        print(len(merge_list))
    return merge_list


def OR(dict_inv, word1, word2):
    list1 = dict_inv[word1]
    list2 = dict_inv[word2]
    not_list = [word for word in list1 if word in list2]
    return not_list


def NOT(dict_inv, word1, word2):
    list1 = dict_inv[word1]
    list2 = dict_inv[word2]
    or_list = [word for word in list1 if word not in list2]
    return or_list


def query_process(query):
    query = data_processing([query])[0]
    query = data_stemming(query)
    return query


if __name__ == '__main__':
    path = 'C:\\Users\\97653\\Desktop\\人工智能\\IR&DM\\Homework3-tweets\\tweets.txt'
    tweets = dict_construction(path)
    dict_inv = build_inverted_index(tweets)
    query = 'Ron Weasley birthday'
    result = merge(dict_inv, query)
    print(result)
