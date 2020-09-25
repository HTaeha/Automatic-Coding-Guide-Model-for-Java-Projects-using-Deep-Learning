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
np.random.seed(1398)
import json, re, nltk, string
import sklearn.metrics as metrics
from nltk.corpus import wordnet
from gensim.models import Word2Vec
from keras.preprocessing import sequence
from keras.models import Model, model_from_json
from keras.layers import Dense, Dropout, Embedding, LSTM, Input, merge, Masking
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

#1. Word2vec parameters
embed_size_word2vec = 200
context_window_word2vec = 5
min_word_frequency_word2vec = int(sys.argv[5])

#2. Input file name
input_file = sys.argv[1]
input_file2 = sys.argv[2]
input_file3 = sys.argv[3]
input_type = sys.argv[4]
input_filename = input_file+'-'+input_type
input_filename2 = input_file2+'-'+input_type
input_filename3 = input_file3+'-'+input_type
java_auto_exception_json = '../MakeJSON/output/Exception/real_final/'+input_filename+'.json'
java_auto_exception_json2 = '../MakeJSON/output/Exception/real_final/'+input_filename2+'.json'
java_auto_exception_json3 = '../MakeJSON/output/Exception/real_final/'+input_filename3+'.json'
model_filename = "all_project-"+input_type+"_frequency"+str(min_word_frequency_word2vec)
count = 1
model_count = 1

#========================================================================================
# Preprocess the java auto exception, extract the vocabulary and learn the word2vec representation
#========================================================================================

with open(java_auto_exception_json,encoding='utf-8-sig') as data_file:           
    data = json.loads(data_file.read(), strict=False)
    
all_data = []
all_exception = []    
all_path = []
all_method = []
for item in data:
    #1. Remove \r 
    current_sentence = item['sentence'].replace('\r', ' ')
    #2. Tokenize
    current_sentence_tokens = nltk.word_tokenize(current_sentence)
    current_sentence_filter = [word.strip(string.punctuation) for word in current_sentence_tokens]
    #3. Join the lists
#current_data = current_sentence_filter
    current_data = current_sentence_tokens
    current_data = list(filter(None, current_data))
    all_data.append(current_data)
    all_exception.append(item['isException'])
    all_path.append(item['path'])
    all_method.append(item['method'])
with open(java_auto_exception_json2,encoding='utf-8-sig') as data_file:           
    data = json.loads(data_file.read(), strict=False)
    
for item in data:
    #1. Remove \r 
    current_sentence = item['sentence'].replace('\r', ' ')
    #2. Tokenize
    current_sentence_tokens = nltk.word_tokenize(current_sentence)
    current_sentence_filter = [word.strip(string.punctuation) for word in current_sentence_tokens]
    #3. Join the lists
#current_data = current_sentence_filter
    current_data = current_sentence_tokens
    current_data = list(filter(None, current_data))
    all_data.append(current_data)
    all_exception.append(item['isException'])
    all_path.append(item['path'])
    all_method.append(item['method'])
with open(java_auto_exception_json3,encoding='utf-8-sig') as data_file:           
    data = json.loads(data_file.read(), strict=False)
    
for item in data:
    #1. Remove \r 
    current_sentence = item['sentence'].replace('\r', ' ')
    #2. Tokenize
    current_sentence_tokens = nltk.word_tokenize(current_sentence)
    current_sentence_filter = [word.strip(string.punctuation) for word in current_sentence_tokens]
    #3. Join the lists
#current_data = current_sentence_filter
    current_data = current_sentence_tokens
    current_data = list(filter(None, current_data))
    all_data.append(current_data)
    all_exception.append(item['isException'])
    all_path.append(item['path'])
    all_method.append(item['method'])

zero = 0
one = 0
for item in all_exception:
    if item == 0 :
        zero = zero + 1
    else :
        one = one +1

print("zero : ")
print(zero)
print("\none : ")
print(one)

under10 =0
over10under200 = 0
over200under400 = 0
over400 = 0

s_sum = 0

for j, curr_row in enumerate(all_data):
    sentence_len_count = 0
    for item in curr_row:
        sentence_len_count += 1
    s_sum += sentence_len_count
    if sentence_len_count <10:
        under10 += 1
    elif sentence_len_count < 200:
        over10under200 += 1
    elif sentence_len_count < 400:
        over200under400 += 1
    else:
        over400 += 1

avg = s_sum/len(all_data)

print(input_filename)
print('all data')
print("\nSentence length Average : %d\n"%(avg))

print("Under 10 : %d"%(under10))
print("Over 10, Under 200 : %d"%(over10under200))
print("Over 200, Under 400 : %d"%(over200under400))
print("Over 400 : %d\n"%(over400))

# Learn the word2vec model and extract vocabulary

wordvec_model = Word2Vec(all_data, min_count=min_word_frequency_word2vec, size=embed_size_word2vec, window=context_window_word2vec)
vocabulary = wordvec_model.wv.vocab
wordvec_model.save("Wordvec_Model/real_final/"+model_filename+"_"+str(count)+".model")

