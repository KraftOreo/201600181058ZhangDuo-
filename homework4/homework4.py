import json
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from math import log


# 获取推特文件
def get_data(path):
    tweets = []
    with open(path, 'r') as f:
        contents = f.readlines()
    for content in contents:
        tweet = json.loads(content)
        tweets.append(tweet)
    print(len(tweets))
    return tweets


# 文本处理
def data_processing(dataset):
    dataset_lower = [data.lower() for data in dataset]
    dataset_tokenized = [word_tokenize(data) for data in dataset_lower]
    stop_words = stopwords.words('english')
    dataset_filtered = [words for words in dataset_tokenized if words not in stop_words]
    english_styled_punctuations = [',', '.', '?', '/', ':', '<', '>', '[', ']', '(', ')', '*', '&', '^', '%', '$', '#',
                                   '@', '!', '~', '`', '+', '_', '-', '“', '”', '~~', '``', '`', '{', '}', '"',
                                   '\\', '…', '\'', '\"', '\'\'', '...', '--', '~~~', 'o', '|', ';']
    dataset_with_no_punctuations = [[words for words in data if words not in english_styled_punctuations] for data in
                                    dataset_filtered]
    return dataset_with_no_punctuations


# 词干化
def data_stemming(dataset):
    ps = PorterStemmer()
    dataset_stemmed = [ps.stem(word) for word in dataset]
    return dataset_stemmed


# 获取tweet文本
def dict_construction(path):
    tweets = get_data(path)
    content = []
    tweetId = []
    for tweet in tweets:
        dataset = [tweet['text'] + tweet['userName'] + tweet['timeStr']]
        tweetid = [tweet['tweetId']]
        tweetId.append(tweetid)
        dataset = data_processing(dataset)
        dataset = dataset[0]
        text = data_stemming(dataset)
        content.append(text)
    return content, tweetId


# 构建inverted_index
def build_inverted_index(tweets):
    dict_inv = {}
    for i in tweets:
        for j in i:
            if j not in dict_inv.keys():
                dict_inv[j] = [tweets.index(i)]
            else:
                dict_inv[j].append(tweets.index(i))
    return dict_inv


# 用and 的方式合并query里每一个词的inverted_index
def merge(dict_inv, query):
    query = query_process(query)
    merge_list = []
    for i in query:
        if i in [*dict_inv.keys()]:
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
    return merge_list[0]


# 用or 的方式合并query里每一个词的inverted_index
def MERGE(dict_inv, query):
    query = query_process(query)
    merge_list = []
    for i in query:
        merge_list.extend(dict_inv[i])
    merge_list = set(merge_list)
    merge_list = list(merge_list)
    return merge_list


# 处理query
def query_process(query):
    query = data_processing([query])[0]
    query = data_stemming(query)
    return query


# 获取query里的词频
def c(word, query):
    count = 0
    for i in query:
        if i == word:
            count += 1
    return count


# 获取大字典里的词频
def df(word, dict_inv, M):
    DF = len(dict_inv[word])
    return DF


# 用pivoted的方法对搜索结果排序取最好的结果
def pivoted_sort_by_similarity(result, query, tweets, dict_inv, tweetid):
    M = len(tweets)
    b = 0.1
    avdl = 0
    for tweet in tweets:
        avdl += len(tweet)
    avdl = avdl / M
    query = query_process(query)
    similarity = {}
    for textnum in result:
        for tweet in tweets:
            f = 0
            d = len(tweets[textnum])
            normalizer = 1 - b + b * d / avdl
            for i in query:
                if i in tweet:
                    f += c(i, query) * log(1 + log(c(i, tweet))) * log((M + 1) / df(i, dict_inv, M)) / normalizer
            similarity[f] = tweetid[textnum][0]
    [*similarity.keys()].sort(reverse=True)
    # ratio = 0.1
    # answer_thres = floor(ratio * len([*similarity.keys()]))
    # answers = [*similarity.keys()][:answer_thres]

    ANSWER = similarity[max([*similarity.keys()])]
    # for answer in answers:
    #     ANSWER.append(similarity[answer])
    return ANSWER


# 用bm25的方法对搜索结果排序取最好的结果
def bm25_sort_by_similarity(result, query, tweets, dict_inv, tweetid):
    M = len(tweets)
    b = 0.1
    avdl = 0
    for tweet in tweets:
        avdl += len(tweet)
    avdl = avdl / M
    query = query_process(query)
    similarity = {}
    k = 100
    for textnum in result:
        for tweet in tweets:
            f = 0
            d = len(tweets[textnum])
            normalizer = 1 - b + b * d / avdl
            for i in query:
                f += (k + 1) * c(i, tweet) * c(i, query) * log((M + 1) / df(i, dict_inv, M)) / (
                        c(i, tweet) + k * normalizer)
            similarity[f] = tweetid[textnum][0]
    [*similarity.keys()].sort()
    # ratio = 0.1
    # answer_thres = floor(ratio * len([*similarity.keys()]))
    # answers = [*similarity.keys()][:answer_thres]
    # ANSWER = []
    # for answer in answers:
    #     ANSWER.append(similarity[answer])
    ANSWER = similarity[max([*similarity.keys()])]

    return ANSWER


# 获取所有query
def get_query():
    contents = []
    with open('Q.txt') as f:
        for line in f:
            content = line.strip().split(' ')
            contents.append(content)
    query = []
    for sentence in contents:
        if sentence[0] == '<query>' and sentence[-1] == '</query>':
            query.append(sentence[1:-1])
    q = []
    for sentence in query:
        sentence = ' '.join(sentence)
        q.append(sentence)
    return q


if __name__ == '__main__':
    # 构建inverted_index并保存起来
    path = 'tweets.txt'
    tweets, tweetid = dict_construction(path)
    # dict_inv = build_inverted_index(tweets)
    # with open('inverted_index.json', 'w') as file_out:
    #     json.dump(dict_inv, file_out, indent=4)
    # 读取字典
    with open('inverted_index.json', 'r') as file_in:
        # content = file_in.readlines()
        dict_inv = json.load(file_in)
    queries = get_query()
    # 对每一个query进行检验并按规定格式输出
    for query in queries:
        result = merge(dict_inv, query)
        print(result)
        if len(result) == 0:
            # result = MERGE(dict_inv, query)
            continue
        similarity = pivoted_sort_by_similarity(result, query, tweets, dict_inv, tweetid)
        with open('pivoted.txt', 'a') as f:
            f.write(str(queries.index(query) + 171) + ' ' + 'Q0' + ' ' + str(similarity) + ' ' + '2' + '\n')
        # answer1 = [str(queries.index(query)) + ' ' + str(sim) + '\n' for sim in similarity]
        # for answer in answer1:
        #     with open('pivoted.txt', 'a') as f:
        #         f.write(answer)
        similarity = bm25_sort_by_similarity(result, query, tweets, dict_inv, tweetid)
        with open('bm25.txt', 'a') as f:
            f.write(str(queries.index(query) + 171) + ' ' + 'Q0' + ' ' + str(similarity) + ' ' + '2' + '\n')
        # answer2 = [str(queries.index(query)) + ' ' + str(sim) + '\n' for sim in similarity]
        # for answer in answer2:
