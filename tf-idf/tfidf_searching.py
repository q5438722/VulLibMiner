import numpy as np
import re
from clean_text import cleaned_text

def get_words_from_object_name(object_name):
    object_name = object_name.strip('/')
    prefix_list = ['maven:' , 'npm:', 'pypi:', 'https://github.com/', 'github.com/']
    for prefix in prefix_list:
        object_name = object_name.replace(prefix, '')
    words = re.split(r'[:|/]', object_name)
    return words

def get_nonzero_count(array):
    return len(array[array > 0])


def get_frequency_single(word, tokens):
    freq = [list(text).count(word) for text in tokens]
    return freq


def get_frequency_multi(words, tokens):
    freq_list = []
    for i, word in enumerate(words):
        freq = get_frequency_single(word, tokens)
        freq_list.append(freq)
    return freq_list


def get_topk_single(scores, repo_urls, topk):
    sorted_id = sorted(range(len(scores)), key=lambda k: scores[k], reverse=True)
    return repo_urls[sorted_id[:topk]]


class TfidfSearching:
    def __init__(self, corpus, topk, ratio):
        """
        :param corpus: dict of lib or repos, should be like {'https://xxx': 'desc'} or {'maven:xxx': 'desc'}
        :param topk:
        :param ratio:
        :param logger:
        """
        # self.logger = logger
        self.topk = topk
        self.ratio = ratio
        self.corpus = corpus
        self.tokens = [word.split() for word in self.corpus['token']]
        self.len_token = [len(t) + 1 for t in self.tokens]

        self.lib_name_index = dict()
        for index, object_name in enumerate(self.corpus['object']):
            core_string = ''.join(get_words_from_object_name(object_name))
            self.lib_name_index[core_string.lower().replace(' ', '')] = (object_name, index)

    def search_topk_objects(self, text_tokens, name_entities=None):
        # self.logger.info('start tfidf searching')
        if len(text_tokens) == 0:
            return []
        if name_entities is None:
            name_entities = []
        search_result = self.search_by_name(name_entities)
        objects_by_ner_name = [name for name, index in search_result]
        objects_by_tfidf = self.get_top_k_based_tfidf(name_entities, text_tokens)

        # topk_objects = objects_by_ner_name
        topk_objects = objects_by_ner_name + [name for name in objects_by_tfidf if name not in objects_by_ner_name]
        # self.logger.info('finish tfidf searching')
        return topk_objects[:self.topk]

    def cal_tf(self, word_freq):
        return word_freq / self.len_token

    def cal_idf(self, word_freq):
        exist_text_count = np.zeros(word_freq.shape)
        for i in range(word_freq.shape[0]):
            exist_text_count[i, :] = get_nonzero_count(word_freq[i, :]) + 1
        return np.log(len(self.len_token) / exist_text_count)

    def cal_tf_idf(self, word_freq, named_entity_index):
        tf = self.cal_tf(word_freq)
        idf = self.cal_idf(word_freq)
        scores = tf * idf
        row_num = np.size(scores, axis=0)
        for index in named_entity_index:
            scores[index] *= 4
            row_num += 3
        return np.sum(scores, axis=0) / row_num

    def get_top_k_based_tfidf(self, ner_key_words, text_tokens):
        word_freq = np.array(get_frequency_multi(text_tokens, self.tokens))
        named_entity_index = []
        if ner_key_words:
            tmp = set(ner_key_words)
            for x in ner_key_words:
                tmp.update(cleaned_text(x))
            named_entity_list = list(tmp)
            named_entity_index = [i for i, token in enumerate(text_tokens) if token in named_entity_list]
        scores = self.cal_tf_idf(word_freq, named_entity_index)

        # self.logger.info('\t average scores, topk rank:start')
        topk_objects = get_topk_single(scores, np.array(self.corpus['object']), self.topk)
        return topk_objects

    def search_by_name(self, named_entity_list):
        res = []
        for named_entity in named_entity_list:
            if named_entity.lower().replace(" ", "") in self.lib_name_index:
                res.append(self.lib_name_index[named_entity.lower().replace(" ", "")])
        return res
