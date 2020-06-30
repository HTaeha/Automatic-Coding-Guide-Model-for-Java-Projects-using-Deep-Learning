# Required dependencies
# 1. NLTK
# 2. Gensim for word2vec
# 3. Keras with tensorflow/theano backend
import random
import sys
import json
import codecs
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import keras
import numpy as np
np.random.seed(1337)
import json, re, nltk, string
import sklearn.metrics as metrics
from nltk.corpus import wordnet
from gensim.models import Word2Vec
from keras.preprocessing import sequence
from keras.models import Model, model_from_json
from keras.layers import Dense, Dropout, Embedding, LSTM, Input, merge
from keras.optimizers import RMSprop
from keras.utils import np_utils
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve, auc

#========================================================================================
# Initializing Hyper parameter
#========================================================================================
# Word2vec parameters
min_word_frequency_word2vec = int(sys.argv[3])

input_file = sys.argv[1]
filetype = sys.argv[2]
filename = input_file + '-' + filetype
model_filename = filename 

model_count = 1

# Load the word2vec model and vocabulary
wordvec_path = "Wordvec_Model/end2/" + model_filename + "_" + str(model_count) + ".model"
wordvec_model = Word2Vec.load(wordvec_path)
vocabulary = wordvec_model.wv.vocab

w2c = dict()
for item in vocabulary:
    w2c[item] = vocabulary[item].count
w2cSorted = sorted(w2c.items(), key=lambda x:x[1], reverse=True)

for i in range(len(w2cSorted)):
    print(w2cSorted[i])
sys.exit(1)

def word_split(curr_data):
    result = []
    for word in curr_data:
        w_len = len(word)
        idx = 0
        while idx < w_len:
            if ord(word[idx]) - ord('0') <= 9 and 0 <= ord(word[idx]) - ord('0'):
                word = word[:idx] + ' ' + word[idx] + ' ' + word[idx+1:]
                idx += 2
                w_len += 2
                continue
            if word[idx] == '_':
                word = word[:idx] + ' ' + word[idx+1:]
                continue
            idx += 1
        w_len = len(word)
        idx = 0
        while idx < w_len:
            if word[idx].isupper():
                if idx == 0:
                    pass
                elif idx == w_len -1:
                    pass
                elif word[idx-1].islower():
                    word = word[:idx] + ' ' + word[idx:]
                    idx += 1
                    w_len += 1
                elif word[idx+1].isupper() or word[idx+1] == ' ':
                    pass
                else:
                    word = word[:idx] + ' ' + word[idx:]
                    idx += 1
                    w_len += 1
            idx += 1
        word_split = word.split()
        for data in word_split:
            result.append(data.lower())
    return result
