import json
import re
import os
import uuid

from nltk.corpus import stopwords
import numpy as np
stopwds = stopwords.words('english')


def expand_apostrophe(string):
    pat_is = re.compile("(it|he|she|that|this|there|here)(\'s)", re.I)
    pat_s = re.compile("(?<=[a-zA-Z])\'s")
    pat_s2 = re.compile("(?<=s)\'s?")
    pat_not = re.compile("(?<=[a-zA-Z])\'t")
    pat_would = re.compile("(?<=[a-zA-Z])\'d")
    pat_will = re.compile("(?<=[a-zA-Z])\'ll")
    pat_am = re.compile("(?<=[I|i])\'m")
    pat_are = re.compile("(?<=[a-zA-Z])\'re")
    pat_ve = re.compile("(?<=[a-zA-Z])\'ve")

    text = pat_is.sub(r"\1 is", string)
    text = re.sub(r"won't", "will not", text)
    text = pat_s.sub("", text)
    text = pat_s2.sub("", text)
    text = pat_not.sub(" not", text)
    text = pat_would.sub(" would", text)
    text = pat_will.sub(" will", text)
    text = pat_am.sub(" am", text)
    text = pat_are.sub(" are", text)
    text = pat_ve.sub(" have", text)
    text = text.replace('\'', ' ')
    return text


def remove_stopwords(tokens):
    return [w for w in tokens if not (w in stopwds)]


def cleaned_text(text):
    text = expand_apostrophe(text)
    text = re.sub(u'[^a-zA-Z]', ' ', text)
    tokens = text.lower().strip().split()
    rmstw_tokens = remove_stopwords(tokens)
    return rmstw_tokens