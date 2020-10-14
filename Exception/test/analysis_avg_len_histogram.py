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

def histogram(input_file, data, vocab, max_sentence_len, x_range, num):
    plt.figure(num)
    hist = [0 for i in range(max_sentence_len//x_range + 1)]
#    hist = []
    for j, curr_row in enumerate(data):
        sentence_len_count = 0
        for item in curr_row:
            if item in vocab:
                sentence_len_count += 1
                if sentence_len_count == max_sentence_len - 1:
                    break
#hist.append(sentence_len_count)
        if sentence_len_count == max_sentence_len - 1:
            hist[-1] += 1
        else:
            hist[sentence_len_count//x_range] += 1
    list_x = []
    for i in range(len(hist)):
        list_x.append(i*x_range)
    plt.bar(list_x, hist, width = 1)
    for i,j in zip(list_x,hist):
        plt.text(i, j, str(j), fontsize = 8)
#plt.hist(hist, histtype = 'bar', rwidth = 0.9, bins = 20)
    plt.title('histogram of '+input_file)
    plt.xlabel('sentence length')
    plt.ylabel('frequency')
    path = 'histogram/real_last/histogram-'+input_file
    plt.savefig(path)

hist_filename = ""

filename = sys.argv[1]
filetype1 = sys.argv[2]
filetype2 = sys.argv[3]
filetype3 = sys.argv[4]
filetype4 = sys.argv[5]
frequency = int(sys.argv[7])
test_filename1 = filename + "-" + filetype1
test_filename2 = filename + "-" + filetype2
test_filename3 = filename + "-" + filetype3
test_filename4 = filename + "-" + filetype4
wordvec_filename1 = test_filename1 +'_frequency'+str(frequency) 
wordvec_filename2 = test_filename2 +'_frequency'+str(frequency) 
wordvec_filename3 = test_filename3 +'_frequency'+str(frequency) 
wordvec_filename4 = test_filename4 +'_frequency'+str(frequency) 
model_filename1 = test_filename1 + '_frequency'+str(frequency)+'_LSTM64_batch64'
model_filename2 = test_filename2 + '_frequency'+str(frequency)+'_LSTM64_batch64'
model_filename3 = test_filename3 + '_frequency'+str(frequency)+'_LSTM64_batch64'
model_filename4 = test_filename4 + '_frequency'+str(frequency)+'_LSTM64_batch64'

test_java_auto_logging_json1 = '../MakeJSON/output/Logging/real_last/'+test_filename1+'.json'
test_java_auto_logging_json2 = '../MakeJSON/output/Logging/real_last/'+test_filename2+'.json'
test_java_auto_logging_json3 = '../MakeJSON/output/Logging/real_last/'+test_filename3+'.json'
test_java_auto_logging_json4 = '../MakeJSON/output/Logging/real_last/'+test_filename4+'.json'

filename = "analysis-hbase"

model_ep = 15
#1, 10, cross-project
model_test_rate = int(sys.argv[6])
model_count = 1

count = 1

#If balance = False, test all data
#If balance = True, test_limit_zero + test_limit_one
balance = True
'''
test_limit_zero = 1767
test_limit_one = 1768
test_limit_one2 = 1154
test_limit_zero2 = 1155
test_limit_one3 = 2033
test_limit_zero3 = 2034
test_limit_one4 = 1809
test_limit_zero4 = 1810
'''

#========================================================================================
# Initializing Hyper parameter
#========================================================================================
#1. Classifier hyperparameters
numCV = 1
min_sentence_len = 10
min_sentence_len2 = 10
min_sentence_len3 = 10
min_sentence_len4 = 10
max_sentence_len = 400
max_sentence_len2 = 400
max_sentence_len3 = 400
max_sentence_len4 = 400
batch_size = 32

embed_size_word2vec = 200

# Load the word2vec model and vocabulary
wordvec_path = "Wordvec_Model/real_last/" + wordvec_filename1 + "_" + str(model_count) + ".model"
wordvec_model = Word2Vec.load(wordvec_path)
vocabulary = wordvec_model.wv.vocab

wordvec_path2 = "Wordvec_Model/real_last/"+ wordvec_filename2 + "_" + str(model_count) + ".model"
wordvec_model2 = Word2Vec.load(wordvec_path2)
vocabulary2 = wordvec_model2.wv.vocab

wordvec_path3 = "Wordvec_Model/real_last/"+ wordvec_filename3 + "_" + str(model_count) + ".model"
wordvec_model3 = Word2Vec.load(wordvec_path3)
vocabulary3 = wordvec_model3.wv.vocab

wordvec_path4 = "Wordvec_Model/real_last/"+ wordvec_filename4 + "_" + str(model_count) + ".model"
wordvec_model4 = Word2Vec.load(wordvec_path4)
vocabulary4 = wordvec_model4.wv.vocab

#Preprocessing test data
with open(test_java_auto_logging_json1,encoding='utf-8-sig') as test_data_file1:
    t_data1 = json.loads(test_data_file1.read(), strict=False)
    
test_data = []
test_logging = []
test_path = []
test_method = []
for item in t_data1:
    #1. Remove \r
    current_sentence = item['sentence'].replace('\r', ' ')
    #2. Tokenize
    current_sentence_tokens = nltk.word_tokenize(current_sentence)
    current_sentence_filter = [word.strip(string.punctuation) for word in current_sentence_tokens]
    #3. Join the lists
    current_data = current_sentence_tokens
    current_data = list(filter(None, current_data))
    test_data.append(current_data)
    test_logging.append(item['isLogged'])
    test_path.append(item['path'])
    test_method.append(item['method'])

#Preprocessing test data2
with open(test_java_auto_logging_json2,encoding='utf-8-sig') as test_data_file2:
    t_data2 = json.loads(test_data_file2.read(), strict=False)
    
test_data2 = []
test_logging2 = []
test_path2 = []
test_method2 = []
for item in t_data2:
    #1. Remove \r
    current_sentence = item['sentence'].replace('\r', ' ')
    #2. Tokenize
    current_sentence_tokens = nltk.word_tokenize(current_sentence)
    current_sentence_filter = [word.strip(string.punctuation) for word in current_sentence_tokens]
    #3. Join the lists
    current_data = current_sentence_tokens
    current_data = list(filter(None, current_data))
    test_data2.append(current_data)
    test_logging2.append(item['isLogged'])
    test_path2.append(item['path'])
    test_method2.append(item['method'])

#Preprocessing test data3
with open(test_java_auto_logging_json3,encoding='utf-8-sig') as test_data_file3:
    t_data3 = json.loads(test_data_file3.read(), strict=False)
    
test_data3 = []
test_logging3 = []
test_path3 = []
test_method3 = []
for item in t_data3:
    #1. Remove \r
    current_sentence = item['sentence'].replace('\r', ' ')
    #2. Tokenize
    current_sentence_tokens = nltk.word_tokenize(current_sentence)
    current_sentence_filter = [word.strip(string.punctuation) for word in current_sentence_tokens]
    #3. Join the lists
    current_data = current_sentence_tokens
    current_data = list(filter(None, current_data))
    test_data3.append(current_data)
    test_logging3.append(item['isLogged'])
    test_path3.append(item['path'])
    test_method3.append(item['method'])
    
#Preprocessing test data4
with open(test_java_auto_logging_json4,encoding='utf-8-sig') as test_data_file4:
    t_data4 = json.loads(test_data_file4.read(), strict=False)
    
test_data4 = []
test_logging4 = []
test_path4 = []
test_method4 = []
for item in t_data4:
    #1. Remove \r
    current_sentence = item['sentence'].replace('\r', ' ')
    #2. Tokenize
    current_sentence_tokens = nltk.word_tokenize(current_sentence)
    current_sentence_filter = [word.strip(string.punctuation) for word in current_sentence_tokens]
    #3. Join the lists
    current_data = current_sentence_tokens
    current_data = list(filter(None, current_data))
    test_data4.append(current_data)
    test_logging4.append(item['isLogged'])
    test_path4.append(item['path'])
    test_method4.append(item['method'])
    
#histogram(hist_filename, test_data, vocabulary, max_sentence_len, 20)

zero = 0
one = 0
for item in test_logging:
	if item == 0:
		zero += 1
	else:
		one += 1
print(test_filename1)
print("zero : ",zero)
print("one : ",one)

zero = 0
one = 0
for item in test_logging2:
	if item == 0:
		zero += 1
	else:
		one += 1
print(test_filename2)
print("zero : ",zero)
print("one : ",one)

zero = 0
one = 0
for item in test_logging3:
	if item == 0:
		zero += 1
	else:
		one += 1
print(test_filename3)
print("zero : ",zero)
print("one : ",one)

zero = 0
one = 0
for item in test_logging4:
	if item == 0:
		zero += 1
	else:
		one += 1
print(test_filename4)
print("zero : ",zero)
print("one : ",one)

'''
for i in range(len(test_data4)):
    if not (test_method[i] == test_method2[i] and test_method2[i] == test_method3[i] and test_method3[i] == test_method4[i]):
        print(test_method[i], test_method2[i], test_method3[i], test_method4[i])
    
'''
#sys.exit(1)
#========================================================================================
# Split cross validation sets and perform deep learning + softmax based classification
#========================================================================================
totalLength = len(test_data)
splitLength = int(totalLength / 10)

totalLength2 = len(test_data2)
splitLength2 = int(totalLength2 / 10)

totalLength3 = len(test_data3)
splitLength3 = int(totalLength3 / 10)

totalLength4 = len(test_data4)
splitLength4 = int(totalLength4 / 10)

for old_index in range(len(test_data)):
    new_index = np.random.randint(len(test_data))
    test_data[old_index], test_data[new_index] = test_data[new_index], test_data[old_index]
    test_logging[old_index], test_logging[new_index] = test_logging[new_index], test_logging[old_index]
    test_path[old_index], test_path[new_index] = test_path[new_index], test_path[old_index]
    test_method[old_index], test_method[new_index] = test_method[new_index], test_method[old_index]
    
    test_data2[old_index], test_data2[new_index] = test_data2[new_index], test_data2[old_index]
    test_logging2[old_index], test_logging2[new_index] = test_logging2[new_index], test_logging2[old_index]
    test_path2[old_index], test_path2[new_index] = test_path2[new_index], test_path2[old_index]
    test_method2[old_index], test_method2[new_index] = test_method2[new_index], test_method2[old_index]
    
    test_data3[old_index], test_data3[new_index] = test_data3[new_index], test_data3[old_index]
    test_logging3[old_index], test_logging3[new_index] = test_logging3[new_index], test_logging3[old_index]
    test_path3[old_index], test_path3[new_index] = test_path3[new_index], test_path3[old_index]
    test_method3[old_index], test_method3[new_index] = test_method3[new_index], test_method3[old_index]
    
    test_data4[old_index], test_data4[new_index] = test_data4[new_index], test_data4[old_index]
    test_logging4[old_index], test_logging4[new_index] = test_logging4[new_index], test_logging4[old_index]
    test_path4[old_index], test_path4[new_index] = test_path4[new_index], test_path4[old_index]
    test_method4[old_index], test_method4[new_index] = test_method4[new_index], test_method4[old_index]

test_data = test_data[(model_test_rate-1)*splitLength:model_test_rate*splitLength-1]
test_logging = test_logging[(model_test_rate-1)*splitLength:model_test_rate*splitLength-1]
test_path = test_path[(model_test_rate-1)*splitLength:model_test_rate*splitLength-1]
test_method = test_method[(model_test_rate-1)*splitLength:model_test_rate*splitLength-1]

test_data2 = test_data2[(model_test_rate-1)*splitLength2:model_test_rate*splitLength2-1]
test_logging2 = test_logging2[(model_test_rate-1)*splitLength2:model_test_rate*splitLength2-1]
test_path2 = test_path2[(model_test_rate-1)*splitLength2:model_test_rate*splitLength2-1]
test_method2 = test_method2[(model_test_rate-1)*splitLength2:model_test_rate*splitLength2-1]

test_data3 = test_data3[(model_test_rate-1)*splitLength3:model_test_rate*splitLength3-1]
test_logging3 = test_logging3[(model_test_rate-1)*splitLength3:model_test_rate*splitLength3-1]
test_path3 = test_path3[(model_test_rate-1)*splitLength3:model_test_rate*splitLength3-1]
test_method3 = test_method3[(model_test_rate-1)*splitLength3:model_test_rate*splitLength3-1]

test_data4 = test_data4[(model_test_rate-1)*splitLength4:model_test_rate*splitLength4-1]
test_logging4 = test_logging4[(model_test_rate-1)*splitLength4:model_test_rate*splitLength4-1]
test_path4 = test_path4[(model_test_rate-1)*splitLength4:model_test_rate*splitLength4-1]
test_method4 = test_method4[(model_test_rate-1)*splitLength4:model_test_rate*splitLength4-1]


# Remove words outside the vocabulary
updated_test_data = []
updated_test_logging = []
updated_test_path = []
updated_test_method = []

updated_test_data2 = []
updated_test_logging2 = []
updated_test_path2 = []
updated_test_method2 = []

updated_test_data3 = []
updated_test_logging3 = []
updated_test_path3 = []
updated_test_method3 = []

updated_test_data4 = []
updated_test_logging4 = []
updated_test_path4 = []
updated_test_method4 = []

for j, item in enumerate(test_data):
    current_test_filter = [word for word in item if word in vocabulary]  
    if len(current_test_filter)>=min_sentence_len:
        updated_test_data.append(current_test_filter)         
        updated_test_logging.append(test_logging[j])
        updated_test_path.append(test_path[j])
        updated_test_method.append(test_method[j])

for j, item in enumerate(test_data2):
    current_test_filter = [word for word in item if word in vocabulary2]
    if len(current_test_filter)>=min_sentence_len2:
        updated_test_data2.append(current_test_filter)         
        updated_test_logging2.append(test_logging2[j])
        updated_test_path2.append(test_path2[j])
        updated_test_method2.append(test_method2[j])
        
for j, item in enumerate(test_data3):
    current_test_filter = [word for word in item if word in vocabulary3]
    if len(current_test_filter)>=min_sentence_len3:
        updated_test_data3.append(current_test_filter)         
        updated_test_logging3.append(test_logging3[j])
        updated_test_path3.append(test_path3[j])
        updated_test_method3.append(test_method3[j])
        
for j, item in enumerate(test_data4):
    current_test_filter = [word for word in item if word in vocabulary4]
    if len(current_test_filter)>=min_sentence_len4:
        updated_test_data4.append(current_test_filter)         
        updated_test_logging4.append(test_logging4[j])
        updated_test_path4.append(test_path4[j])
        updated_test_method4.append(test_method4[j])
test_one = 0
test_zero = 0
test_one2 = 0
test_zero2 = 0
test_one3 = 0
test_zero3 = 0
test_one4 = 0
test_zero4 = 0
for i, data in enumerate(updated_test_logging):
    if data == 1:
        test_one += 1
    else:
        test_zero += 1
for i, data in enumerate(updated_test_logging2):
    if data == 1:
        test_one2 += 1
    else:
        test_zero2 += 1
for i, data in enumerate(updated_test_logging3):
    if data == 1:
        test_one3 += 1
    else:
        test_zero3 += 1
for i, data in enumerate(updated_test_logging4):
    if data == 1:
        test_one4 += 1
    else:
        test_zero4 += 1
        
if balance == True:
#'''
    test_limit_one = min(test_one, test_zero)
    test_limit_zero = test_limit_one
    test_limit_one2 = min(test_one2, test_zero2)
    test_limit_zero2 = test_limit_one2
    test_limit_one3 = min(test_one3, test_zero3)
    test_limit_zero3 = test_limit_one3
    test_limit_one4 = min(test_one4, test_zero4)
    test_limit_zero4 = test_limit_one4
#'''
    final_test_data = []
    final_test_logging = []
    final_test_path = []
    final_test_method = []

    final_test_data2 = []
    final_test_logging2 = []
    final_test_path2 = []
    final_test_method2 = []

    final_test_data3 = []
    final_test_logging3 = []
    final_test_path3 = []
    final_test_method3 = []
    
    final_test_data4 = []
    final_test_logging4 = []
    final_test_path4 = []
    final_test_method4 = []
    
    count_zero = 0
    count_one = 0
    for i, item in enumerate(updated_test_logging):
        if item == 0 and count_zero <= test_limit_zero:
            final_test_data.append(updated_test_data[i])
            final_test_logging.append(updated_test_logging[i])
            final_test_path.append(updated_test_path[i])
            final_test_method.append(updated_test_method[i])
        
            count_zero += 1
        elif item == 1 and count_one <= test_limit_one:
            final_test_data.append(updated_test_data[i])
            final_test_logging.append(updated_test_logging[i])
            final_test_path.append(updated_test_path[i])
            final_test_method.append(updated_test_method[i])
           
            count_one += 1
        if count_zero == test_limit_zero and count_one == test_limit_one:
            break

    count_zero = 0
    count_one = 0
    for i, item in enumerate(updated_test_logging2):
        if item == 0 and count_zero <= test_limit_zero2:
            final_test_data2.append(updated_test_data2[i])
            final_test_logging2.append(updated_test_logging2[i])
            final_test_path2.append(updated_test_path2[i])
            final_test_method2.append(updated_test_method2[i])
            
            count_zero += 1
        elif item == 1 and count_one <= test_limit_one2:
            final_test_data2.append(updated_test_data2[i])
            final_test_logging2.append(updated_test_logging2[i])
            final_test_path2.append(updated_test_path2[i])
            final_test_method2.append(updated_test_method2[i])
            
            count_one += 1
        if count_zero == test_limit_zero2 and count_one == test_limit_one2:
            break
            
    count_zero = 0
    count_one = 0
    for i, item in enumerate(updated_test_logging3):
        if item == 0 and count_zero <= test_limit_zero3:
            final_test_data3.append(updated_test_data3[i])
            final_test_logging3.append(updated_test_logging3[i])
            final_test_path3.append(updated_test_path3[i])
            final_test_method3.append(updated_test_method3[i])
            
            count_zero += 1
        elif item == 1 and count_one <= test_limit_one3:
            final_test_data3.append(updated_test_data3[i])
            final_test_logging3.append(updated_test_logging3[i])
            final_test_path3.append(updated_test_path3[i])
            final_test_method3.append(updated_test_method3[i])
            
            count_one += 1
        if count_zero == test_limit_zero3 and count_one == test_limit_one3:
            break
            
    count_zero = 0
    count_one = 0
    for i, item in enumerate(updated_test_logging4):
        if item == 0 and count_zero <= test_limit_zero4:
            final_test_data4.append(updated_test_data4[i])
            final_test_logging4.append(updated_test_logging4[i])
            final_test_path4.append(updated_test_path4[i])
            final_test_method4.append(updated_test_method4[i])
            
            count_zero += 1
        elif item == 1 and count_one <= test_limit_one4:
            final_test_data4.append(updated_test_data4[i])
            final_test_logging4.append(updated_test_logging4[i])
            final_test_path4.append(updated_test_path4[i])
            final_test_method4.append(updated_test_method4[i])
            
            count_one += 1
        if count_zero == test_limit_zero4 and count_one == test_limit_one4:
            break
else:
    final_test_data = updated_test_data
    final_test_logging = updated_test_logging
    final_test_path = updated_test_path
    final_test_method = updated_test_method

    final_test_data2 = updated_test_data2
    final_test_logging2 = updated_test_logging2
    final_test_path2 = updated_test_path2
    final_test_method2 = updated_test_method2
    
    final_test_data3 = updated_test_data3
    final_test_logging3 = updated_test_logging3
    final_test_path3 = updated_test_path3
    final_test_method3 = updated_test_method3
    
    final_test_data4 = updated_test_data4
    final_test_logging4 = updated_test_logging4
    final_test_path4 = updated_test_path4
    final_test_method4 = updated_test_method4


X_test = np.empty(shape=[len(final_test_data), max_sentence_len, embed_size_word2vec], dtype='float32')
Y_test = np.empty(shape=[len(final_test_logging),1], dtype='int32')
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
    Y_test[j,0] = final_test_logging[j]
        
X_test2 = np.empty(shape=[len(final_test_data2), max_sentence_len2, embed_size_word2vec], dtype='float32')
Y_test2 = np.empty(shape=[len(final_test_logging2), 1], dtype='int32')
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
    Y_test2[j,0] = final_test_logging2[j]

X_test3 = np.empty(shape=[len(final_test_data3), max_sentence_len3, embed_size_word2vec], dtype='float32')
Y_test3 = np.empty(shape=[len(final_test_logging3), 1], dtype='int32')
# 1 - start of sentence, # 2 - end of sentence, # 0 - zero padding. Hence, word indices start with 3 
for j, curr_row in enumerate(final_test_data3):
    sequence_cnt = 0          
    for item in curr_row:
        if item in vocabulary3:
            X_test3[j, sequence_cnt, :] = wordvec_model3[item]
            sequence_cnt = sequence_cnt + 1                
            if sequence_cnt == max_sentence_len3-1:
                break                
    for k in range(sequence_cnt, max_sentence_len3):
        X_test3[j, k, :] = np.zeros((1,embed_size_word2vec))
    Y_test3[j,0] = final_test_logging3[j]

X_test4 = np.empty(shape=[len(final_test_data4), max_sentence_len4, embed_size_word2vec], dtype='float32')
Y_test4 = np.empty(shape=[len(final_test_logging4), 1], dtype='int32')
# 1 - start of sentence, # 2 - end of sentence, # 0 - zero padding. Hence, word indices start with 3 
for j, curr_row in enumerate(final_test_data4):
    sequence_cnt = 0          
    for item in curr_row:
        if item in vocabulary4:
            X_test4[j, sequence_cnt, :] = wordvec_model4[item]
            sequence_cnt = sequence_cnt + 1                
            if sequence_cnt == max_sentence_len4-1:
                break                
    for k in range(sequence_cnt, max_sentence_len4):
        X_test4[j, k, :] = np.zeros((1,embed_size_word2vec))
    Y_test4[j,0] = final_test_logging4[j]

# Load model
model_json = "Model/real_last/"+str(model_test_rate)+"/best/JAL_"+ model_filename1+"_"+str(model_count)+"_model.json"
model_h5 = "Model/real_last/"+str(model_test_rate)+"/best/JAL_"+ model_filename1+"_"+str(model_count)+"_model.h5"
json_file = open(model_json, "r")
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights(model_h5)

model_json2 = "Model/real_last/"+str(model_test_rate)+"/best/JAL_"+ model_filename2+"_"+str(model_count)+"_model.json"
model_h5_2 = "Model/real_last/"+str(model_test_rate)+"/best/JAL_"+ model_filename2+"_"+str(model_count)+"_model.h5"
json_file2 = open(model_json2, "r")
loaded_model_json2 = json_file2.read()
json_file2.close()
model2 = model_from_json(loaded_model_json2)
model2.load_weights(model_h5_2)

model_json3 = "Model/real_last/"+str(model_test_rate)+"/best/JAL_"+ model_filename3+"_"+str(model_count)+"_model.json"
model_h5_3 = "Model/real_last/"+str(model_test_rate)+"/best/JAL_"+ model_filename3+"_"+str(model_count)+"_model.h5"
json_file3 = open(model_json3, "r")
loaded_model_json3 = json_file3.read()
json_file3.close()
model3 = model_from_json(loaded_model_json3)
model3.load_weights(model_h5_3)

model_json4 = "Model/real_last/"+str(model_test_rate)+"/best/JAL_"+ model_filename4+"_"+str(model_count)+"_model.json"
model_h5_4 = "Model/real_last/"+str(model_test_rate)+"/best/JAL_"+ model_filename4+"_"+str(model_count)+"_model.h5"
json_file4 = open(model_json4, "r")
loaded_model_json4 = json_file4.read()
json_file4.close()
model4 = model_from_json(loaded_model_json4)
model4.load_weights(model_h5_4)

print("Loaded model from disk")

rms = RMSprop(lr = 0.001, rho = 0.9, epsilon = 1e-08)
model.compile(loss = 'categorical_crossentropy', optimizer = rms, metrics = ['accuracy'])
model2.compile(loss = 'categorical_crossentropy', optimizer = rms, metrics = ['accuracy'])
model3.compile(loss = 'categorical_crossentropy', optimizer = rms, metrics = ['accuracy'])
model4.compile(loss = 'categorical_crossentropy', optimizer = rms, metrics = ['accuracy'])
predict = model.predict(X_test)
predict2 = model2.predict(X_test2)
predict3 = model3.predict(X_test3)
predict4 = model4.predict(X_test4)

preds = predict[:,1]
preds2 = predict2[:,1]
preds3 = predict3[:,1]
preds4 = predict4[:,1]

fpr, tpr, threshold = roc_curve(final_test_logging, preds)
fpr2, tpr2, threshold2 = roc_curve(final_test_logging2, preds2)
fpr3, tpr3, threshold3 = roc_curve(final_test_logging3, preds3)
fpr4, tpr4, threshold4 = roc_curve(final_test_logging4, preds4)

roc_auc = auc(fpr, tpr)
roc_auc2 = auc(fpr2, tpr2)
roc_auc3 = auc(fpr3, tpr3)
roc_auc4 = auc(fpr4, tpr4)

predictY=[]
for k in predict:
    predictY.append(list(k).index(max(k)))
predictY2=[]
for m in predict2:
    predictY2.append(list(m).index(max(m)))
predictY3=[]
for m in predict3:
    predictY3.append(list(m).index(max(m)))
predictY4=[]
for m in predict4:
    predictY4.append(list(m).index(max(m)))

print(test_filename1)
print(test_filename2)
print(test_filename3)
print(test_filename4)
print()

label_log = []
label_not_log = []
log_sum = 0
not_log_sum = 0
for i, data in enumerate(final_test_logging):
    if data == 0:
        label_not_log.append(final_test_data[i])
        not_log_sum += len(final_test_data[i])
    else:
        label_log.append(final_test_data[i])
        log_sum += len(final_test_data[i])
log_avg = log_sum/len(label_log)
not_log_avg = not_log_sum/len(label_not_log)
print('log_sum', log_sum)
print('not_log_sum', not_log_sum)
print('log', len(label_log))
print('not log', len(label_not_log))

print('log_avg', log_avg)
print('not_log_avg',not_log_avg)
print()
log_histogram_filename1 = 'Label-Logged-'+wordvec_filename1
not_log_histogram_filename1 = 'Label-Not_Logged-'+wordvec_filename1
histogram(log_histogram_filename1,label_log,vocabulary,max_sentence_len,20,1)
histogram(not_log_histogram_filename1,label_not_log,vocabulary,max_sentence_len,20,2)

label_log = []
label_not_log = []
log_sum = 0
not_log_sum = 0
for i, data in enumerate(final_test_logging2):
    if data == 0:
        label_not_log.append(final_test_data2[i])
        not_log_sum += len(final_test_data2[i])
    else:
        label_log.append(final_test_data2[i])
        log_sum += len(final_test_data2[i])
log_avg = log_sum/len(label_log)
not_log_avg = not_log_sum/len(label_not_log)
print('log_sum', log_sum)
print('not_log_sum', not_log_sum)
print('log', len(label_log))
print('not log', len(label_not_log))

print('log_avg', log_avg)
print('not_log_avg',not_log_avg)
print()
log_histogram_filename2 = 'Label-Logged-'+wordvec_filename2
not_log_histogram_filename2 = 'Label-Not_Logged-'+wordvec_filename2
histogram(log_histogram_filename2,label_log,vocabulary2,max_sentence_len,20,3)
histogram(not_log_histogram_filename2,label_not_log,vocabulary2,max_sentence_len,20,4)

label_log = []
label_not_log = []
log_sum = 0
not_log_sum = 0
for i, data in enumerate(final_test_logging3):
    if data == 0:
        label_not_log.append(final_test_data3[i])
        not_log_sum += len(final_test_data3[i])
    else:
        label_log.append(final_test_data3[i])
        log_sum += len(final_test_data3[i])
log_avg = log_sum/len(label_log)
not_log_avg = not_log_sum/len(label_not_log)
print('log_sum', log_sum)
print('not_log_sum', not_log_sum)
print('log', len(label_log))
print('not log', len(label_not_log))

print('log_avg', log_avg)
print('not_log_avg',not_log_avg)
print()
log_histogram_filename3 = 'Label-Logged-'+wordvec_filename3
not_log_histogram_filename3 = 'Label-Not_Logged-'+wordvec_filename3
histogram(log_histogram_filename3,label_log,vocabulary3,max_sentence_len,20,5)
histogram(not_log_histogram_filename3,label_not_log,vocabulary3,max_sentence_len,20,6)


label_log = []
label_not_log = []
log_sum = 0
not_log_sum = 0
for i, data in enumerate(final_test_logging4):
    if data == 0:
        label_not_log.append(final_test_data4[i])
        not_log_sum += len(final_test_data4[i])
    else:
        label_log.append(final_test_data4[i])
        log_sum += len(final_test_data4[i])
log_avg = log_sum/len(label_log)
not_log_avg = not_log_sum/len(label_not_log)
print('log_sum', log_sum)
print('not_log_sum', not_log_sum)
print('log', len(label_log))
print('not log', len(label_not_log))

print('log_avg', log_avg)
print('not_log_avg',not_log_avg)
print()
log_histogram_filename4 = 'Label-Logged-'+wordvec_filename4
not_log_histogram_filename4 = 'Label-Not_Logged-'+wordvec_filename4
histogram(log_histogram_filename4,label_log,vocabulary4,max_sentence_len,20,7)
histogram(not_log_histogram_filename4,label_not_log,vocabulary4,max_sentence_len,20,8)


'''
predict_log = []
predict_not_log = []
log_sum = 0
not_log_sum = 0
for i, data in enumerate(predictY):
    if data == 0:
        predict_not_log.append(final_test_data[i])
        not_log_sum += len(final_test_data[i])
    else:
        predict_log.append(final_test_data[i])
        log_sum += len(final_test_data[i])
log_avg = log_sum/len(predict_log)
not_log_avg = not_log_sum/len(predict_not_log)
print('log_sum', log_sum)
print('not_log_sum', not_log_sum)
print('log', len(predict_log))
print('not log', len(predict_not_log))

print('log_avg', log_avg)
print('not_log_avg',not_log_avg)
print()
log_histogram_filename1 = 'Logged-'+wordvec_filename1
not_log_histogram_filename1 = 'Not_Logged-'+wordvec_filename1
histogram(log_histogram_filename1,predict_log,vocabulary,max_sentence_len,20,1)
histogram(not_log_histogram_filename1,predict_not_log,vocabulary,max_sentence_len,20,2)

predict_log = []
predict_not_log = []
log_sum = 0
not_log_sum = 0
for i, data in enumerate(predictY2):
    if data == 0:
        predict_not_log.append(final_test_data2[i])
        not_log_sum += len(final_test_data2[i])
    else:
        predict_log.append(final_test_data2[i])
        log_sum += len(final_test_data2[i])
log_avg = log_sum/len(predict_log)
not_log_avg = not_log_sum/len(predict_not_log)
print('log_sum', log_sum)
print('not_log_sum', not_log_sum)
print('log', len(predict_log))
print('not log', len(predict_not_log))

print('log_avg', log_avg)
print('not_log_avg',not_log_avg)

print()
log_histogram_filename2 = 'Logged-'+wordvec_filename2
not_log_histogram_filename2 = 'Not_Logged-'+wordvec_filename2
histogram(log_histogram_filename2,predict_log,vocabulary2,max_sentence_len,20,3)
histogram(not_log_histogram_filename2,predict_not_log,vocabulary2,max_sentence_len,20,4)

predict_log = []
predict_not_log = []
log_sum = 0
not_log_sum = 0
for i, data in enumerate(predictY3):
    if data == 0:
        predict_not_log.append(final_test_data3[i])
        not_log_sum += len(final_test_data3[i])
    else:
        predict_log.append(final_test_data3[i])
        log_sum += len(final_test_data3[i])
log_avg = log_sum/len(predict_log)
not_log_avg = not_log_sum/len(predict_not_log)
print('log_sum', log_sum)
print('not_log_sum', not_log_sum)
print('log', len(predict_log))
print('not log', len(predict_not_log))

print('log_avg', log_avg)
print('not_log_avg',not_log_avg)

print()
log_histogram_filename3 = 'Logged-'+wordvec_filename3
not_log_histogram_filename3 = 'Not_Logged-'+wordvec_filename3
histogram(log_histogram_filename3,predict_log,vocabulary3,max_sentence_len,20,5)
histogram(not_log_histogram_filename3,predict_not_log,vocabulary3,max_sentence_len,20,6)

predict_log = []
predict_not_log = []
log_sum = 0
not_log_sum = 0
for i, data in enumerate(predictY4):
    if data == 0:
        predict_not_log.append(final_test_data4[i])
        not_log_sum += len(final_test_data4[i])
    else:
        predict_log.append(final_test_data4[i])
        log_sum += len(final_test_data4[i])
log_avg = log_sum/len(predict_log)
not_log_avg = not_log_sum/len(predict_not_log)
print('log_sum', log_sum)
print('not_log_sum', not_log_sum)
print('log', len(predict_log))
print('not log', len(predict_not_log))

print('log_avg', log_avg)
print('not_log_avg',not_log_avg)

print()
log_histogram_filename4 = 'Logged-'+wordvec_filename4
not_log_histogram_filename4 = 'Not_Logged-'+wordvec_filename4
histogram(log_histogram_filename4,predict_log,vocabulary4,max_sentence_len,20,7)
histogram(not_log_histogram_filename4,predict_not_log,vocabulary4,max_sentence_len,20,8)
'''
test_one = 0
test_zero = 0
choose_one = 0
choose_zero = 0
idx = 0
true = 0
for i, data in enumerate(final_test_logging):
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

f1_score = metrics.f1_score(final_test_logging, predictY)
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
for i, data in enumerate(final_test_logging2):
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

f1_score = metrics.f1_score(final_test_logging2, predictY2)
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
    
test_one = 0
test_zero = 0
choose_one = 0
choose_zero = 0
idx = 0
true = 0
for i, data in enumerate(final_test_logging3):
    if data == 1:
        test_one += 1
    else:
        test_zero += 1
for i, data in enumerate(predictY3):
    if data == 1:
        choose_one = choose_one + 1
    if data == Y_test3[idx][0]:
        true = true +1
    idx = idx + 1
for i, data in enumerate(predictY3):
    if data == 0:
        choose_zero = choose_zero + 1
accuracy = (float(true)/len(predict3))*100

print("\nTest accuracy:", accuracy)
print("\ntest_zero : ", test_zero)
print("\ntest_one : ", test_one)
print("\nchoose_zero : ", choose_zero)
print("\nchoose_one : ", choose_one)

f1_score = metrics.f1_score(final_test_logging3, predictY3)
print("\nF1score : ", f1_score)

print("\nAUC : " + str(roc_auc3))


print("\nConfusion Matrix3")
print(confusion_matrix(Y_test3, predictY3))
cm3 = confusion_matrix(Y_test3, predictY3)
cm3 = cm3.astype('float') / cm3.sum(axis=1)[:, np.newaxis]
for p in range(cm3.shape[0]):
    print('True label', p)
    for q in range(cm3.shape[0]):
        print(cm3[p,q], end=' ')
        if q%100 == 0:
            print(' ')
    print(' ')
    
test_one = 0
test_zero = 0
choose_one = 0
choose_zero = 0
idx = 0
true = 0
for i, data in enumerate(final_test_logging4):
    if data == 1:
        test_one += 1
    else:
        test_zero += 1
for i, data in enumerate(predictY4):
    if data == 1:
        choose_one = choose_one + 1
    if data == Y_test4[idx][0]:
        true = true +1
    idx = idx + 1
for i, data in enumerate(predictY4):
    if data == 0:
        choose_zero = choose_zero + 1
accuracy = (float(true)/len(predict4))*100

print("\nTest accuracy:", accuracy)
print("\ntest_zero : ", test_zero)
print("\ntest_one : ", test_one)
print("\nchoose_zero : ", choose_zero)
print("\nchoose_one : ", choose_one)

f1_score = metrics.f1_score(final_test_logging4, predictY4)
print("\nF1score : ", f1_score)

print("\nAUC : " + str(roc_auc4))


print("\nConfusion Matrix4")
print(confusion_matrix(Y_test4, predictY4))
cm4 = confusion_matrix(Y_test4, predictY4)
cm4 = cm4.astype('float') / cm4.sum(axis=1)[:, np.newaxis]
for p in range(cm4.shape[0]):
    print('True label', p)
    for q in range(cm4.shape[0]):
        print(cm4[p,q], end=' ')
        if q%100 == 0:
            print(' ')
    print(' ')
'''
plt.figure(1)
plt.title('hbase-AST')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.4f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0,1], [0,1], 'r--')
plt.xlim([0,1])
plt.ylim([0,1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
roc_name = "Analysis/ROC/"+str(model_test_rate)+"/"+test_filename1+"_"+str(model_count)
plt.savefig(roc_name)

plt.figure(2)
plt.title('hbase-code')
plt.plot(fpr2, tpr2, 'b', label = 'AUC = %0.4f' % roc_auc2)
plt.legend(loc = 'lower right')
plt.plot([0,1], [0,1], 'r--')
plt.xlim([0,1])
plt.ylim([0,1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
roc_name2 = "Analysis/ROC/"+str(model_test_rate)+"/"+test_filename2+"_"+str(model_count)
plt.savefig(roc_name2)

plt.figure(3)
plt.title('hbase-code')
plt.plot(fpr3, tpr3, 'b', label = 'AUC = %0.4f' % roc_auc3)
plt.legend(loc = 'lower right')
plt.plot([0,1], [0,1], 'r--')
plt.xlim([0,1])
plt.ylim([0,1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
roc_name3 = "Analysis/ROC/"+str(model_test_rate)+"/"+test_filename3+"_"+str(model_count)
plt.savefig(roc_name3)

plt.figure(4)
plt.title('hbase-code')
plt.plot(fpr4, tpr4, 'b', label = 'AUC = %0.4f' % roc_auc4)
plt.legend(loc = 'lower right')
plt.plot([0,1], [0,1], 'r--')
plt.xlim([0,1])
plt.ylim([0,1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
roc_name4 = "Analysis/ROC/"+str(model_test_rate)+"/"+test_filename4+"_"+str(model_count)
plt.savefig(roc_name4)
'''
