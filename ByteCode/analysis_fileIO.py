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

filename = sys.argv[1]
filetype1 = "AST"
filetype2 = sys.argv[2]
test_file1 = filename + "-" + filetype1
test_file2 = filename + "-" + filetype2
model_filename = filename + "-function_AST_balanced_max400_masking"
model_filename2 = filename + "-original_code_balanced_max400_masking"

#test_java_auto_exception_json = '../MakeJSON/output/ByteCode/'+test_file1+'.json'
test_java_auto_exception_json2 = '../MakeJSON/output/ByteCode/'+test_file2+'.json'

model_ep = 15
#1, 10, cross-project
model_test_rate = 10
model_count = 1

count = 1

#If balance = False, test all data
#If balance = True, test_limit_zero + test_limit_one
balance = False
test_limit_zero = 1100
test_limit_one = 1100
test_limit_one2 = 1707
test_limit_zero2 = 1706


#========================================================================================
# Initializing Hyper parameter
#========================================================================================
#1. Classifier hyperparameters
numCV = 1
min_sentence_len = 10
min_sentence_len2 = 10
max_sentence_len = 400
max_sentence_len2 = 400
batch_size = 32

#2. Word2vec parameters
min_word_frequency_word2vec = 5
embed_size_word2vec = 200
context_window_word2vec = 5
'''
# Load the word2vec model and vocabulary
wordvec_path = "Wordvec_Model/" + str(model_test_rate) + "/" + model_filename + "_min" + str(min_sentence_len) + "_" + str(model_count) + ".model"
wordvec_model = Word2Vec.load(wordvec_path)
vocabulary = wordvec_model.wv.vocab
'''
'''
wordvec_path2 = "Wordvec_Model/" + str(model_test_rate) + "/" + model_filename2 + "_min" + str(min_sentence_len2) + "_" + str(model_count) + ".model"
#wordvec_path2 = "Wordvec_Model/final/" + str(model_test_rate) + "/" + test_file2 + "_" + str(model_count) + ".model"
wordvec_model2 = Word2Vec.load(wordvec_path2)
vocabulary2 = wordvec_model2.wv.vocab

w2c = dict()
for item in vocabulary2:
    w2c[item] = vocabulary2[item].count
w2cSorted = sorted(w2c.items(), key=lambda x:x[1], reverse=True)
'''
'''
for i in range(len(w2cSorted)):
    print(w2cSorted[i])
print(w2cSorted)
'''
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
'''
#Preprocessing test data
with open(test_java_auto_exception_json,encoding='utf-8-sig') as test_data_file:
    t_data = json.loads(test_data_file.read(), strict=False)
    
test_data = []
test_exception = []
test_path = []
test_method = []
for item in t_data:
    #1. Remove \r
    current_sentence = item['sentence'].replace('\r', ' ')
    #2. Tokenize
    current_sentence_tokens = nltk.word_tokenize(current_sentence)
    current_sentence_filter = [word.strip(string.punctuation) for word in current_sentence_tokens]
    #3. Join the lists
    current_data = current_sentence_tokens
    current_data = list(filter(None, current_data))
    test_data.append(current_data)
    test_exception.append(item['isException'])
    test_path.append(item['path'])
    test_method.append(item['method'])
'''
#Preprocessing test data2
with open(test_java_auto_exception_json2,encoding='utf-8-sig') as test_data_file2:
    t_data2 = json.loads(test_data_file2.read(), strict=False)
    
test_data2 = []
test_exception2 = []
test_path2 = []
test_method2 = []
no_punctuation_data = []
for item in t_data2:
    #1. Remove \r
    current_sentence = item['sentence'].replace('\r', ' ')
    #2. Tokenize
    current_sentence_tokens = nltk.word_tokenize(current_sentence)
    current_sentence_filter = [word.strip(string.punctuation) for word in current_sentence_tokens]
    #3. Join the lists
    current_data = current_sentence_tokens
    current_data = list(filter(None, current_data))
#low_data = word_split(current_data)
#test_data2.append(low_data)
    test_data2.append(current_data)
    test_exception2.append(item['isException'])
    test_path2.append(item['path'])
    test_method2.append(item['method'])

    temp = current_sentence_filter
    temp = list(filter(None, temp))
#lowercase_data = []
#   for word in temp:
#lowercase_data.append(word.lower())
#lowercase_data = word_split(temp)
#no_punctuation_data.append(lowercase_data)
    no_punctuation_data.append(temp)

wv_model2 = Word2Vec(test_data2, min_count = min_word_frequency_word2vec, size = embed_size_word2vec, window = context_window_word2vec)
vocabulary2 = wv_model2.wv.vocab

wv_model = Word2Vec(no_punctuation_data, min_count = min_word_frequency_word2vec, size = embed_size_word2vec, window = context_window_word2vec)
vocab = wv_model.wv.vocab
w2c_no_punc = dict()
for item in vocab:
    w2c_no_punc[item] = vocab[item].count
w2cSorted_no_punc = sorted(w2c_no_punc.items(), key=lambda x:x[1], reverse=True)
'''
for i in range(len(w2cSorted_no_punc)):
    print(w2cSorted_no_punc[i])
sys.exit(1)
'''
#========================================================================================
# Split cross validation sets and perform deep learning + softmax based classification
#========================================================================================
'''
totalLength = len(test_data)
splitLength = int(totalLength / 10)
'''
totalLength2 = len(test_data2)
splitLength2 = int(totalLength2 / 10)

for old_index in range(len(test_data2)):
    new_index = np.random.randint(old_index+1)
    '''
    test_data[old_index], test_data[new_index] = test_data[new_index], test_data[old_index]
    test_exception[old_index], test_exception[new_index] = test_exception[new_index], test_exception[old_index]
    test_path[old_index], test_path[new_index] = test_path[new_index], test_path[old_index]
    test_method[old_index], test_method[new_index] = test_method[new_index], test_method[old_index]
    '''
    test_data2[old_index], test_data2[new_index] = test_data2[new_index], test_data2[old_index]
    test_exception2[old_index], test_exception2[new_index] = test_exception2[new_index], test_exception2[old_index]
    test_path2[old_index], test_path2[new_index] = test_path2[new_index], test_path2[old_index]
    test_method2[old_index], test_method2[new_index] = test_method2[new_index], test_method2[old_index]
'''
test_data = test_data[(model_test_rate-1)*splitLength:model_test_rate*splitLength-1]
test_exception = test_exception[(model_test_rate-1)*splitLength:model_test_rate*splitLength-1]
test_path = test_path[(model_test_rate-1)*splitLength:model_test_rate*splitLength-1]
test_method = test_method[(model_test_rate-1)*splitLength:model_test_rate*splitLength-1]

test_data2 = test_data2[(model_test_rate-1)*splitLength2:model_test_rate*splitLength2-1]
test_exception2 = test_exception2[(model_test_rate-1)*splitLength2:model_test_rate*splitLength2-1]
test_path2 = test_path2[(model_test_rate-1)*splitLength2:model_test_rate*splitLength2-1]
test_method2 = test_method2[(model_test_rate-1)*splitLength2:model_test_rate*splitLength2-1]
'''

# Remove words outside the vocabulary
'''
updated_test_data = []
updated_test_exception = []
updated_test_path = []
updated_test_method = []
'''
updated_test_data2 = []
updated_test_exception2 = []
updated_test_path2 = []
updated_test_method2 = []

'''
for j, item in enumerate(test_data):
    current_test_filter = [word for word in item if word in vocabulary]  
    if len(current_test_filter)>=min_sentence_len:
        updated_test_data.append(current_test_filter)         
        updated_test_exception.append(test_exception[j])
        updated_test_path.append(test_path[j])
        updated_test_method.append(test_method[j])
'''
for j, item in enumerate(test_data2):
    current_test_filter = [word for word in item if word in vocabulary2]
    if len(current_test_filter)>=min_sentence_len2:
        updated_test_data2.append(current_test_filter)         
        updated_test_exception2.append(test_exception2[j])
        updated_test_path2.append(test_path2[j])
        updated_test_method2.append(test_method2[j])

test_one = 0
test_zero = 0
test_one2 = 0
test_zero2 = 0
'''
for i, data in enumerate(updated_test_exception):
    if data == 1:
        test_one += 1
    else:
        test_zero += 1
'''
for i, data in enumerate(updated_test_exception2):
    if data == 1:
        test_one2 += 1
    else:
        test_zero2 += 1

if balance == True:
    '''
    test_limit_one = min(test_one, test_zero)
    test_limit_zero = test_limit_one
    test_limit_one2 = min(test_one2, test_zero2)
    test_limit_zero2 = test_limit_one2
    '''
    '''
    final_test_data = []
    final_test_exception = []
    final_test_path = []
    final_test_method = []
    '''
    final_test_data2 = []
    final_test_exception2 = []
    final_test_path2 = []
    final_test_method2 = []

    count_zero = 0
    count_one = 0
    '''
    for i, item in enumerate(updated_test_exception):
        if item == 0 and count_zero <= test_limit_zero:
            final_test_data.append(updated_test_data[i])
            final_test_exception.append(updated_test_exception[i])
            final_test_path.append(updated_test_path[i])
            final_test_method.append(updated_test_method[i])
        
            count_zero += 1
        elif item == 1 and count_one <= test_limit_one:
            final_test_data.append(updated_test_data[i])
            final_test_exception.append(updated_test_exception[i])
            final_test_path.append(updated_test_path[i])
            final_test_method.append(updated_test_method[i])
           
            count_one += 1
        if count_zero == test_limit_zero and count_one == test_limit_one:
            break
    '''
    count_zero = 0
    count_one = 0
    for i, item in enumerate(updated_test_exception2):
        if item == 0 and count_zero <= test_limit_zero2:
            final_test_data2.append(updated_test_data2[i])
            final_test_exception2.append(updated_test_exception2[i])
            final_test_path2.append(updated_test_path2[i])
            final_test_method2.append(updated_test_method2[i])
            
            count_zero += 1
        elif item == 1 and count_one <= test_limit_one2:
            final_test_data2.append(updated_test_data2[i])
            final_test_exception2.append(updated_test_exception2[i])
            final_test_path2.append(updated_test_path2[i])
            final_test_method2.append(updated_test_method2[i])
            
            count_one += 1
        if count_zero == test_limit_zero2 and count_one == test_limit_one2:
            break
else:
    '''
    final_test_data = updated_test_data
    final_test_exception = updated_test_exception
    final_test_path = updated_test_path
    final_test_method = updated_test_method
    '''
    final_test_data2 = updated_test_data2
    final_test_exception2 = updated_test_exception2
    final_test_path2 = updated_test_path2
    final_test_method2 = updated_test_method2

'''
X_test = np.empty(shape=[len(final_test_data), max_sentence_len, embed_size_word2vec], dtype='float32')
Y_test = np.empty(shape=[len(final_test_exception),1], dtype='int32')
# 1 - start of sentence, # 2 - end of sentence, # 0 - zero padding. Hence, word indices start with 3 
for j, curr_row in enumerate(final_test_data):
    sequence_cnt = 0          
    for item in curr_row:
        if item in vocabulary:
            X_test[j, sequence_cnt, :] = wordvec_model[item]
            sequence_cnt = sequence_cnt + 1                
            if sequence_cnt == max_sentence_len-1:
                break                
    for k in range(sequence_cnt, max_sentence_len):
        X_test[j, k, :] = np.zeros((1,embed_size_word2vec))
    Y_test[j,0] = final_test_exception[j]
        
X_test2 = np.empty(shape=[len(final_test_data2), max_sentence_len2, embed_size_word2vec], dtype='float32')
Y_test2 = np.empty(shape=[len(final_test_exception2), 1], dtype='int32')
# 1 - start of sentence, # 2 - end of sentence, # 0 - zero padding. Hence, word indices start with 3 
for j, curr_row in enumerate(final_test_data2):
    sequence_cnt = 0          
    for item in curr_row:
        if item in vocabulary2:
            X_test2[j, sequence_cnt, :] = wordvec_model2[item]
            sequence_cnt = sequence_cnt + 1                
            if sequence_cnt == max_sentence_len2-1:
                break                
    for k in range(sequence_cnt, max_sentence_len2):
        X_test2[j, k, :] = np.zeros((1,embed_size_word2vec))
    Y_test2[j,0] = final_test_exception2[j]

# Load model
model_json = "Model/"+str(model_test_rate)+"/JAE_"+ model_filename+"_ep"+str(model_ep)+"_min"+str(min_sentence_len)+"_"+str(model_count)+"_model.json"
model_h5 = "Model/"+str(model_test_rate)+"/JAE_"+ model_filename+"_ep"+str(model_ep)+"_min"+str(min_sentence_len)+"_"+str(model_count)+"_model.h5"
json_file = open(model_json, "r")
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights(model_h5)

model_json2 = "Model/"+str(model_test_rate)+"/JAE_"+ model_filename2+"_ep"+str(model_ep)+"_min"+str(min_sentence_len)+"_"+str(model_count)+"_model.json"
model_h52 = "Model/"+str(model_test_rate)+"/JAE_"+ model_filename2+"_ep"+str(model_ep)+"_min"+str(min_sentence_len)+"_"+str(model_count)+"_model.h5"
json_file2 = open(model_json2, "r")
loaded_model_json2 = json_file2.read()
json_file2.close()
model2 = model_from_json(loaded_model_json2)
model2.load_weights(model_h52)

print("Loaded model from disk")

rms = RMSprop(lr = 0.001, rho = 0.9, epsilon = 1e-08)
model.compile(loss = 'categorical_crossentropy', optimizer = rms, metrics = ['accuracy'])
model2.compile(loss = 'categorical_crossentropy', optimizer = rms, metrics = ['accuracy'])
predict = model.predict(X_test)
predict2 = model2.predict(X_test2)

preds = predict[:,1]
preds2 = predict2[:,1]

fpr, tpr, threshold = roc_curve(final_test_exception, preds)
fpr2, tpr2, threshold2 = roc_curve(final_test_exception2, preds2)

roc_auc = auc(fpr, tpr)
roc_auc2 = auc(fpr2, tpr2)

predictY=[]
for k in predict:
    predictY.append(list(k).index(max(k)))
predictY2=[]
for m in predict2:
    predictY2.append(list(m).index(max(m)))

print(test_file1)
print(test_file2)
print()

AST1_code1 = 0
AST1_code0 = 0
AST0_code1 = 0
AST0_code0 = 0

ASTT_codeT = 0
ASTT_codeF = 0
ASTF_codeT = 0
ASTF_codeF = 0

fileT_AST1_code1 = 0
fileT_AST1_code0 = 0
fileT_AST0_code1 = 0
fileT_AST0_code0 = 0

fileF_AST1_code1 = 0
fileF_AST1_code0 = 0
fileF_AST0_code1 = 0
fileF_AST0_code0 = 0

fileT_ASTT_codeT = 0
fileT_ASTT_codeF = 0
fileT_ASTF_codeT = 0
fileT_ASTF_codeF = 0

fileF_ASTT_codeT = 0
fileF_ASTT_codeF = 0
fileF_ASTF_codeT = 0
fileF_ASTF_codeF = 0
'''
fileT_code1 = 0
fileF_code1 = 0
fileT_code0 = 0
fileF_code0 = 0

file_flag = False
exception_keyword = []
'''
for i in range(100):
    exception_keyword.append(w2cSorted_no_punc[i][0])
    print(w2cSorted_no_punc[i][0])

'''
ratio = int(sys.argv[3])
for i in range(100):
    if len(w2cSorted_no_punc) <= i:
        break
    count1 = 0
    count0 = 0
    print(i+1)
    print(w2cSorted_no_punc[i][0])
    for j, data in enumerate(final_test_exception2):
        for idx in range(len(final_test_data2[j])):
            if w2cSorted_no_punc[i][0] in final_test_data2[j][idx]:
                if data == 1:
                    count1 += 1
                else:
                    count0 += 1
                break
    print("Label 1 : ", count1)
    print("Label 0 : ", count0)
    if count1 >= ratio*count0:
        exception_keyword.append(w2cSorted_no_punc[i][0])
        print("keyword!")
    print()
print()
print("Exception keyword : ", exception_keyword)

word_count = 0
limit_count = int(sys.argv[4])
for i, data in enumerate(final_test_exception2):
    word_count = 0
    if data == 1:
        for idx in range(len(final_test_data2[i])):
            for a, keyword in enumerate(exception_keyword):
                if keyword in final_test_data2[i][idx]:
                    word_count += 1
                    if word_count >= limit_count:
                        fileT_code1 += 1
                        file_flag = True
                        break
            if file_flag:
                break
        if file_flag:
            file_flag = False
        else:
             fileF_code1 += 1
    else:
        for idx in range(len(final_test_data2[i])):
            for a, keyword in enumerate(exception_keyword):
                if keyword in final_test_data2[i][idx]:
                   word_count += 1
                   if word_count >= limit_count:
                        fileT_code0 += 1
                        file_flag = True
                        break
            if file_flag:
                break
        if file_flag:
            file_flag = False
        else:
             fileF_code0 += 1

'''
file_flag = False
for i, data in enumerate(final_test_exception):
    if predictY[i] == 1:
        for i2, data2 in enumerate(final_test_exception2):
            if final_test_path[i] == final_test_path2[i2] and final_test_method[i] == final_test_method2[i2]:
                if predictY2[i2] == 1:
                    AST1_code1 += 1
                    for idx in range(len(final_test_data2[i2])):
                        for a, keyword in enumerate(exception_keyword):
                            if keyword in final_test_data2[i2][idx].lower():
                                fileT_AST1_code1 += 1
                                file_flag = True
                                break
                        if file_flag:
                            break
                    if file_flag:
                        file_flag = False
                    else:
                        fileF_AST1_code1 += 1
                else:
                    AST1_code0 += 1
                    for idx in range(len(final_test_data2[i2])):
                        for a, keyword in enumerate(exception_keyword):
                            if keyword in final_test_data2[i2][idx].lower():
                                fileT_AST1_code0 += 1
                                file_flag = True
                                break
                        if file_flag:
                            break
                    if file_flag:
                        file_flag = False
                    else:
                        fileF_AST1_code0 += 1
                break
    else:
        for i2, data2 in enumerate(final_test_exception2):
            if final_test_path[i] == final_test_path2[i2] and final_test_method[i] == final_test_method2[i2]:
                if predictY2[i2] == 1:
                    AST0_code1 += 1
                    for idx in range(len(final_test_data2[i2])):
                        for a, keyword in enumerate(exception_keyword):
                            if keyword in final_test_data2[i2][idx].lower():
                                fileT_AST0_code1 += 1
                                file_flag = True
                                break
                        if file_flag:
                            break
                    if file_flag:
                        file_flag = False
                    else:
                        fileF_AST0_code1 += 1
                else:
                    AST0_code0 += 1
                    for idx in range(len(final_test_data2[i2])):
                        for a, keyword in enumerate(exception_keyword):
                            if keyword in final_test_data2[i2][idx].lower():
                                fileT_AST0_code0 += 1
                                file_flag = True
                                break
                        if file_flag:
                            break
                    if file_flag:
                        file_flag = False
                    else:
                        fileF_AST0_code0 += 1
                break
    if predictY[i] == Y_test[i][0]:
        for i2, data2 in enumerate(final_test_exception2):
            if final_test_path[i] == final_test_path2[i2] and final_test_method[i] == final_test_method2[i2]:
                if predictY2[i2] == Y_test2[i2][0]:
                    ASTT_codeT += 1
                    for idx in range(len(final_test_data2[i2])):
                        for a, keyword in enumerate(exception_keyword):
                            if keyword in final_test_data2[i2][idx].lower():
                                fileT_ASTT_codeT += 1
                                file_flag = True
                                break
                        if file_flag:
                            break
                    if file_flag:
                        file_flag = False
                    else:
                        fileF_ASTT_codeT += 1
                else:
                    ASTT_codeF += 1
                    for idx in range(len(final_test_data2[i2])):
                        for a, keyword in enumerate(exception_keyword):
                            if keyword in final_test_data2[i2][idx].lower():
                                fileT_ASTT_codeF += 1
                                file_flag = True
                                break
                        if file_flag:
                            break
                    if file_flag:
                        file_flag = False
                    else:
                        fileF_ASTT_codeF += 1
                break
    else:
        for i2, data2 in enumerate(final_test_exception2):
            if final_test_path[i] == final_test_path2[i2] and final_test_method[i] == final_test_method2[i2]:
                if predictY2[i2] == Y_test2[i2][0]:
                    ASTF_codeT += 1
                    for idx in range(len(final_test_data2[i2])):
                        for a, keyword in enumerate(exception_keyword):
                            if keyword in final_test_data2[i2][idx].lower():
                                fileT_ASTF_codeT += 1
                                file_flag = True
                                break
                        if file_flag:
                            break
                    if file_flag:
                        file_flag = False
                    else:
                        fileF_ASTF_codeT += 1
                else:
                    ASTF_codeF += 1
                    for idx in range(len(final_test_data2[i2])):
                        for a, keyword in enumerate(exception_keyword):
                            if keyword in final_test_data2[i2][idx].lower():
                                fileT_ASTF_codeF += 1
                                file_flag = True
                                break
                        if file_flag:
                            break
                    if file_flag:
                        file_flag = False
                    else:
                        fileF_ASTF_codeF += 1
                break
'''
                     
print("Label : 1, file exist. ", fileT_code1)
print("Label : 1, file doesn't exist. ", fileF_code1)
print("Label : 0, file exist. ", fileT_code0)
print("Label : 0, file doesn't exist. ", fileF_code0)
print()
'''
print("AST1_code1 : "+str(AST1_code1))
print("AST1_code0 : "+str(AST1_code0))
print("AST0_code1 : "+str(AST0_code1))
print("AST0_code0 : "+str(AST0_code0))
print()
print("ASTT_codeT : ", ASTT_codeT)
print("ASTT_codeF : ", ASTT_codeF)
print("ASTF_codeT : ", ASTF_codeT)
print("ASTF_codeF : ", ASTF_codeF)
print()
print("fileT_ASTT_codeT : ", fileT_ASTT_codeT)
print("fileT_ASTT_codeF : ", fileT_ASTT_codeF)
print("fileT_ASTF_codeT : ", fileT_ASTF_codeT)
print("fileT_ASTF_codeF : ", fileT_ASTF_codeF)
print()
print("fileF_ASTT_codeT : ", fileF_ASTT_codeT)
print("fileF_ASTT_codeF : ", fileF_ASTT_codeF)
print("fileF_ASTF_codeT : ", fileF_ASTF_codeT)
print("fileF_ASTF_codeF : ", fileF_ASTF_codeF)
print()
print("fileT_AST1_code1 : ", fileT_AST1_code1)
print("fileT_AST1_code0 : ", fileT_AST1_code0)
print("fileT_AST0_code1 : ", fileT_AST0_code1)
print("fileT_AST0_code0 : ", fileT_AST0_code0)
print()
print("fileF_AST1_code1 : ", fileF_AST1_code1)
print("fileF_AST1_code0 : ", fileF_AST1_code0)
print("fileF_AST0_code1 : ", fileF_AST0_code1)
print("fileF_AST0_code0 : ", fileF_AST0_code0)
print()


test_one = 0
test_zero = 0
choose_one = 0
choose_zero = 0
idx = 0
true = 0
for i, data in enumerate(final_test_exception):
    if data == 1:
        test_one += 1
    else:
        test_zero += 1
for i, data in enumerate(predictY):
    if data == 1:
        choose_one = choose_one + 1
    if data == Y_test[idx][0]:
        true = true +1
    idx = idx + 1
for i, data in enumerate(predictY):
    if data == 0:
        choose_zero = choose_zero + 1
accuracy = (float(true)/len(predict))*100

print("\nTest accuracy:", accuracy)
print("\ntest_zero : ", test_zero)
print("\ntest_one : ", test_one)
print("\nchoose_zero : ", choose_zero)
print("\nchoose_one : ", choose_one)

f1_score = metrics.f1_score(final_test_exception, predictY)
print("\nF1score : ", f1_score)

print("\nAUC : " + str(roc_auc))

    
print("\nConfusion Matrix1")
print(confusion_matrix(Y_test, predictY))
cm = confusion_matrix(Y_test, predictY)
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
for p in range(cm.shape[0]):
    print('True label', p)
    for q in range(cm.shape[0]):
        print(cm[p,q], end=' ')
        if q%100 == 0:
            print(' ')
    print(' ')


test_one = 0
test_zero = 0
choose_one = 0
choose_zero = 0
idx = 0
true = 0
for i, data in enumerate(final_test_exception2):
    if data == 1:
        test_one += 1
    else:
        test_zero += 1
for i, data in enumerate(predictY2):
    if data == 1:
        choose_one = choose_one + 1
    if data == Y_test2[idx][0]:
        true = true +1
    idx = idx + 1
for i, data in enumerate(predictY2):
    if data == 0:
        choose_zero = choose_zero + 1
accuracy = (float(true)/len(predict2))*100

print("\nTest accuracy:", accuracy)
print("\ntest_zero : ", test_zero)
print("\ntest_one : ", test_one)
print("\nchoose_zero : ", choose_zero)
print("\nchoose_one : ", choose_one)

f1_score = metrics.f1_score(final_test_exception2, predictY2)
print("\nF1score : ", f1_score)

print("\nAUC : " + str(roc_auc2))


print("\nConfusion Matrix2")
print(confusion_matrix(Y_test2, predictY2))
cm2 = confusion_matrix(Y_test2, predictY2)
cm2 = cm2.astype('float') / cm2.sum(axis=1)[:, np.newaxis]
for p in range(cm2.shape[0]):
    print('True label', p)
    for q in range(cm2.shape[0]):
        print(cm2[p,q], end=' ')
        if q%100 == 0:
            print(' ')
    print(' ')
'''
