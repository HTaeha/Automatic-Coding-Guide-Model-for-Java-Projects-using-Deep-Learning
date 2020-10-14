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

import keras
import numpy as np
np.random.seed(1398)
import json, re, nltk, string
import sklearn.metrics as metrics
from gensim.models import Word2Vec
from keras.utils import np_utils

def remove_values_from_list(the_list, val):
    return [value for value in the_list if value != val]

def open_data(json_file, label_name):
    with open(json_file,encoding='utf-8-sig') as data_file:           
        data = json.loads(data_file.read(), strict=False)
        
    all_data = []
    all_label = []    
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
        current_data = remove_values_from_list(current_data, ';')
        all_data.append(current_data)
        all_label.append(item[label_name])
        all_path.append(item['path'])
        all_method.append(item['method'])

    return all_data, all_label, all_path, all_method

def print_the_number_of_data_split_document_size(data):
    under10 =0
    over10under30 = 0
    over30under100 = 0
    over100under150 = 0
    over150under200 = 0
    over200under400 = 0
    over400 = 0

    s_sum = 0

    for j, curr_row in enumerate(data):
        sentence_len_count = 0
        for item in curr_row:
            sentence_len_count += 1
        s_sum += sentence_len_count
        if sentence_len_count <10:
            under10 += 1
        elif sentence_len_count <30:
            over10under30 += 1
        elif sentence_len_count <100:
            over30under100 += 1
        elif sentence_len_count <150:
            over100under150 += 1
        elif sentence_len_count < 200:
            over150under200 += 1
        elif sentence_len_count < 400:
            over200under400 += 1
        else:
            over400 += 1
    avg = s_sum/len(data)

    print("\nSentence length Average : %d\n"%(avg))
    print("Under 10 : %d"%(under10))
    print("Over 10, Under 30 : %d"%(over10under30))
    print("Over 30, Under 100 : %d"%(over30under100))
    print("Over 100, Under 150 : %d"%(over100under150))
    print("Over 150, Under 200 : %d"%(over150under200))
    print("Over 200, Under 400 : %d"%(over200under400))
    print("Over 400 : %d\n"%(over400))

def train_word2vec_model(data, min_word_frequency_word2vec, embed_size_word2vec, context_window_word2vec, model_name):
    wordvec_model = Word2Vec(data, min_count=min_word_frequency_word2vec, size=embed_size_word2vec, window=context_window_word2vec)
    vocabulary = wordvec_model.wv.vocab
    wordvec_model.save(model_name)
    return wordvec_model, vocabulary

def load_word2vec_model(model_name):
    wordvec_path = model_name
    wordvec_model = Word2Vec.load(wordvec_path)
    vocabulary = wordvec_model.wv.vocab
    return wordvec_model, vocabulary

#========================================================================================
# Split cross validation sets and perform deep learning + softmax based classification
#========================================================================================
def data_shuffle(data, seed_num):
    np.random.seed(seed_num)
    for old_index in range(len(data)):
        new_index = np.random.randint(len(data))
        data[old_index], data[new_index] = data[new_index], data[old_index]

def numpy_data_shuffle(data, seed_num):
    np.random.seed(seed_num)
    for old_index in range(len(data)):
        new_index = np.random.randint(len(data))
        data[old_index], data[new_index] = data[new_index], data[old_index].copy()

def split_train_test_data(test_rate, data, cross_validation_num):
    totalLength = len(data)
    splitLength = int(totalLength / cross_validation_num)

    if test_rate == 1:
        train_data = data[splitLength:]
        test_data = data[:splitLength-1]
    else: 
        train_data = data[:(test_rate-1)*splitLength-1] + data[test_rate*splitLength:]
        test_data = data[(test_rate-1)*splitLength:test_rate*splitLength-1]

    return train_data, test_data

def random_train_test_data(data, train_num, seed_num):
    random.seed(seed_num)
    random_idx = random.sample(range(0, len(data)), train_num)
    train_data = []
    test_data = []
    for i, item in enumerate(data):
        if i in random_idx:
            train_data.append(item)
        else:
            test_data.append(item)

    return train_data, test_data

def remove_words_outside_the_vocabulary(data, vocabulary):
    updated_data = []
    for j, item in enumerate(data):
        current_train_filter = [word for word in item if word in vocabulary]
        updated_data.append(current_train_filter)
    return updated_data

def set_document_size(data, label_data, path_data, method_data, min_len, max_len):
    updated_data = []
    updated_label = []
    updated_path = []
    updated_method = []
    for i, sentence in enumerate(data):
        sentence_len = len(sentence)
        if min_len <= sentence_len and sentence_len < max_len:
            updated_data.append(sentence)
            updated_label.append(label_data[i])
            updated_path.append(path_data[i])
            updated_method.append(method_data[i])
    return updated_data, updated_label, updated_path, updated_method

def set_document_size2(standard_data, data, label_data, path_data, method_data, min_len, max_len):
    updated_data = []
    updated_label = []
    updated_path = []
    updated_method = []
    for i, sentence in enumerate(standard_data):
        sentence_len = len(sentence)
        if min_len <= sentence_len and sentence_len < max_len:
            updated_data.append(data[i])
            updated_label.append(label_data[i])
            updated_path.append(path_data[i])
            updated_method.append(method_data[i])
    return updated_data, updated_label, updated_path, updated_method

def balance_out_data(data, label_data, path_data, method_data, limit_num):
    balanced_data = []
    balanced_label = []
    balanced_path = []
    balanced_method = []
    count_zero = 0
    count_one = 0
    for i, item in enumerate(label_data):
        if item == 0 and count_zero <= limit_num:
            balanced_data.append(data[i])
            balanced_label.append(label_data[i])
            balanced_path.append(path_data[i])
            balanced_method.append(method_data[i])
            count_zero += 1
        elif item == 1 and count_one <= limit_num:
            balanced_data.append(data[i])
            balanced_label.append(label_data[i])
            balanced_path.append(path_data[i])
            balanced_method.append(method_data[i])
            count_one += 1
        if count_zero == limit_num and count_one == limit_num:
            break
    return balanced_data, balanced_label, balanced_path, balanced_method

def make_input_data_json(input_data_path, label_name, data, label, path, method):
    len_check = len(data)
    with open(input_data_path, 'w', encoding="utf-8-sig") as f:
        f.write('[\r\n')
        for i, item in enumerate(data):
            f.write('\t{\r\n')
            f.write('\t\t\"sentence\" : \"')
            for j, item2 in enumerate(item):
                f.write(item2+' ')
            f.write("\",\r\n")
            f.write("\t\t\""+label_name+"\" : "+str(label[i])+",\r\n")
            f.write("\t\t\"path\" : \""+path[i]+"\",\r\n")
            f.write("\t\t\"method\" : \""+method[i]+"\"\r\n")
            if len_check == i+1:
                f.write("\t}\r\n")
            else:
                f.write("\t},\r\n")
        f.write("]")

def make_input_merge_data_json(input_data_path, label_name, merge_data, label, path, method):
    len_check = len(merge_data[0])
    with open(input_data_path, 'w', encoding="utf-8-sig") as f:
        f.write('[\r\n')
        for i, item in enumerate(merge_data[0]):
            f.write('\t{\r\n')
            for num in range(len(merge_data)):
                f.write('\t\t\"sentence'+str(num+1)+'\" : \"')
                for j, item2 in enumerate(merge_data[num][i]):
                    f.write(item2+' ')
                f.write("\",\r\n")
            f.write("\t\t\""+label_name+"\" : "+str(label[i])+",\r\n")
            f.write("\t\t\"path\" : \""+path[i]+"\",\r\n")
            f.write("\t\t\"method\" : \""+method[i]+"\"\r\n")
            if len_check == i+1:
                f.write("\t}\r\n")
            else:
                f.write("\t},\r\n")
        f.write("]")

def check_max_sentence_len(data, path, method):
    max_len = 0
    max_idx = 0
    for i, item in enumerate(data):
        if max_len < len(item):
            max_idx = i
        max_len = max(max_len, len(item))
#print(data[max_idx], max_len, path[max_idx], method[max_idx])
    return max_len

# Create data for deep learning + softmax
def create_deep_learning_input_data(data, label_data, max_sentence_len, embed_size_word2vec, vocabulary, wordvec_model):
    X = np.empty(shape=[len(data), max_sentence_len, embed_size_word2vec], dtype='float32')
    Y = np.empty(shape=[len(label_data),1], dtype='int32')
    for j, curr_row in enumerate(data):
        sequence_cnt = 0         
        for item in curr_row:
            if item in vocabulary:
                X[j, sequence_cnt, :] = wordvec_model[item] 
                sequence_cnt += 1                
                if sequence_cnt == max_sentence_len-1:
                    break                
        for k in range(sequence_cnt, max_sentence_len):
            X[j, k, :] = np.zeros((1,embed_size_word2vec))        
        Y[j,0] = label_data[j]
    Y = np_utils.to_categorical(Y, 2)
    return X, Y
