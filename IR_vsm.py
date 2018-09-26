import os
import sys
import chardet
import codecs
from bs4 import BeautifulSoup
from gensim import corpora, models, similarities
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


def get_encoding(file):
    # 二进制方式读取，获取字节数据，检测类型
    with open(file, 'rb') as f:
        return chardet.detect(f.read())['encoding']


def Decode_Content(contents):
    new_contents = []
    for content in contents:
        content = content.decode()
        new_contents.append(content)
    return new_contents


# of no need
def change_the_suffixName(path0, path1):
    sys.path.append(path1)
    files = os.listdir(path0)
    for filename in files:
        portion = os.path.splitext(filename)
        # 如果后缀是.txt
        if portion[1] == "":
            # 重新组合文件名和后缀名
            newname = portion[0] + ".html"
            filenamedir = path1 + filename
            newnamedir = path1 + newname
            # os.rename(filename,newname)
            os.rename(filenamedir, newnamedir)


def read_and_decode_content(path):
    new_contents = []
    files = os.listdir(path)
    for file in files:
        with open(path + '\\' + file, 'rb') as f:
            contents = f.readlines()
            code_format = get_encoding(path + '\\' + file)
            if not code_format == None:
                for content in contents:
                    content = content.decode(code_format)
                    new_contents.append(content)
    return new_contents


# If you would like to change the encoding format to utf-8, use the function
def convert_file_to_utf8(filename):
    # !!! does not backup the origin file
    content = codecs.open(filename, 'r').read()
    source_encoding = chardet.detect(content)['encoding']
    if source_encoding == None:
        print("??", filename)
        return
    print("  ", source_encoding, filename)
    if source_encoding != 'utf-8' and source_encoding != 'UTF-8-SIG':
        content = content.decode(source_encoding, 'ignore')  # .encode(source_encoding)
        codecs.open(filename, 'w', encoding='UTF-8-SIG').write(content)


def convert_file_to_html(filename):
    # !!! does not backup the origin file
    content = codecs.open(filename, 'rb').read()
    source_encoding = chardet.detect(content)['encoding']
    if source_encoding == None:
        print("??", filename)
        return
    print("  ", source_encoding, filename)
    if source_encoding != 'html':
        content = content.decode(source_encoding, 'ignore')  # .encode(source_encoding)
        codecs.open(filename, 'w', encoding='UTF-8').write(content)


def build_dataset(path):
    dirs = os.listdir(path)
    contents = []
    for dir in dirs:
        files = os.listdir(path + '\\' + dir)
        for file in files:
            with open(path + '\\' + dir + '\\' + file, 'rb') as text:
                all_contents = text.readlines()
                code_format = get_encoding(path + '\\' + dir + '\\' + file)
                if not code_format == None:
                    for content in all_contents:
                        decoded_content = content.decode(code_format)
                        decoded_content = str(decoded_content)
                        contents.append(decoded_content)
    return contents


def build_classification_dataset(path):
    contents = []
    files = os.listdir(path)
    for file in files:
        with open(path + '\\' + file, 'rb') as text:
            all_contents = text.readlines()
            code_format = get_encoding(path + '\\' + file)
            if not code_format == None:
                for content in all_contents:
                    decoded_content = content.decode(code_format)
                    decoded_content = str(decoded_content)
                    contents.append(decoded_content)
    return contents


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

    return contents


def wash_data(dataset):
    new_dataset = []
    for data in dataset:
        if data[-1] == '\n':
            data = data[::-1]
        if data == '\n':
            del (data)
            continue
        new_dataset.append(data)

    return new_dataset


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
    tfidf_model = models.TfidfModel(corpus, normalize=True, smartirs='ltn')
    corpus_tfidf = [tfidf_model[doc] for doc in corpus]
    return corpus_tfidf, dictionary, corpus


def tfidf_model_small(dataset, dir):
    corpus_tfidf, dictionary, corpus = build_vs_model(dataset)
    return {'corpus_tfidf_small' + dir: corpus_tfidf, 'dictionary_small' + dir: dictionary,
            'corpus_small' + dir: corpus}


if __name__ == '__main__':
    path = "C:\\Users\\97653\\Desktop\\人工智能\\20news-19997\\20_newsgroups"
    dataset = wash_data(build_dataset(path))
    dict = build_vs_model(dataset)
    with open('Results1.txt', 'a') as f:
        f.write(dict)
    for dir in os.listdir(path):
        files = os.listdir(path + '\\' + dir)
        for file in files:
            dataset = wash_data(build_small_dataset(path + '\\' + dir + '\\' + file))
            vsm = tfidf_model_small(dataset, file)
            with open('Results2.txt', 'a') as f:
                f.write(vsm, '\n')

        dataset = wash_data(build_classification_dataset(path + '\\' + dir))
        dict2 = build_vs_model(dataset)
        with open('Results3.txt', 'a')as f:
            f.write(dict2,'\n')