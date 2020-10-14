# Required dependencies
# 1. NLTK
# 2. Gensim for word2vec
# 3. Keras with tensorflow/theano backend
import random
import sys
import json
from collections import OrderedDict
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
import pickle

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
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.externals import joblib

#========================================================================================
# Initializing Hyper parameter
#========================================================================================

#1. Classifier hyperparameters
numCV = 1
min_sentence_len = 10
max_sentence_len = 400

#2. Word2vec parameters
embed_size_word2vec = 200
context_window_word2vec = 5
min_word_frequency_word2vec = int(sys.argv[4])

#3. Input file name
input_file = sys.argv[1]
input_type = sys.argv[2]
input_filename = input_file+'-'+input_type
java_auto_logging_json = '../MakeJSON/output/Logging/real_final/'+input_filename+'.json'
filename = input_file+"-DecisionTree-"+input_type+"_frequency"+str(min_word_frequency_word2vec)
#filename = input_filename+"_combine_wordvec_frequency"+str(min_word_frequency_word2vec)
model_filename = input_filename+"_frequency"+str(min_word_frequency_word2vec)
#model_filename = "all_project-"+input_type+"_frequency"+str(min_word_frequency_word2vec)
epoch_len = 15
test_rate = int(sys.argv[3])
count = 1
model_count = 1

#========================================================================================
# Preprocess the java auto logging, extract the vocabulary and learn the word2vec representation
#========================================================================================

with open(java_auto_logging_json,encoding='utf-8-sig') as data_file:           
    data = json.loads(data_file.read(), strict=False)
    
all_data = []
all_logging = []    
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
    all_logging.append(item['isLogged'])
    all_path.append(item['path'])
    all_method.append(item['method'])

zero = 0
one = 0
for item in all_logging:
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

'''
wordvec_model = Word2Vec(all_data, min_count=min_word_frequency_word2vec, size=embed_size_word2vec, window=context_window_word2vec)
vocabulary = wordvec_model.wv.vocab
wordvec_model.save("Wordvec_Model/real_final/"+model_filename+"_"+str(count)+".model")
sys.exit(1)
'''
#'''
wordvec_path = "../JavaAutoLogging/Wordvec_Model/real_final/"+model_filename+"_"+str(model_count)+".model"
wordvec_model = Word2Vec.load(wordvec_path)
vocabulary = wordvec_model.wv.vocab
#'''
#========================================================================================
# Split cross validation sets and perform deep learning + softmax based classification
#========================================================================================
totalLength = len(all_data)
splitLength = int(totalLength / 10)
#'''
for old_index in range(len(all_data)):
    new_index = np.random.randint(len(all_data))
    all_data[old_index], all_data[new_index] = all_data[new_index], all_data[old_index]
    all_logging[old_index], all_logging[new_index] = all_logging[new_index], all_logging[old_index]
    all_path[old_index], all_path[new_index] = all_path[new_index], all_path[old_index]
    all_method[old_index], all_method[new_index] = all_method[new_index], all_method[old_index]
#'''
# Split cross validation set

if test_rate == 1:
    train_data = all_data[splitLength:]
    test_data = all_data[:splitLength-1]
    train_logging = all_logging[splitLength:]
    test_logging = all_logging[:splitLength-1]
    train_path = all_path[splitLength:]
    test_path = all_path[:splitLength]
    train_method = all_method[splitLength:]
    test_method = all_method[:splitLength]
else: 
    train_data = all_data[:(test_rate-1)*splitLength-1] + all_data[test_rate*splitLength:]
    test_data = all_data[(test_rate-1)*splitLength:test_rate*splitLength-1]
    train_logging = all_logging[:(test_rate-1)*splitLength-1]+ all_logging[test_rate*splitLength:]
    test_logging = all_logging[(test_rate-1)*splitLength:test_rate*splitLength-1]
    train_path = all_path[:(test_rate-1)*splitLength-1]+ all_path[test_rate*splitLength:]
    test_path = all_path[(test_rate-1)*splitLength:test_rate*splitLength-1]
    train_method = all_method[:(test_rate-1)*splitLength-1]+ all_method[test_rate*splitLength:]
    test_method = all_method[(test_rate-1)*splitLength:test_rate*splitLength-1]


# Remove words outside the vocabulary
updated_train_data = []    
updated_train_logging = []
updated_train_path = []
updated_train_method = []
updated_test_data = []
updated_test_logging = []
updated_test_path = []
updated_test_method = []

for j, item in enumerate(train_data):
    current_train_filter = [word for word in item if word in vocabulary]
    if len(current_train_filter)>=min_sentence_len:
        updated_train_data.append(current_train_filter)
        updated_train_logging.append(train_logging[j])  
        updated_train_path.append(train_path[j])
        updated_train_method.append(train_method[j])

for j, item in enumerate(test_data):
    current_test_filter = [word for word in item if word in vocabulary]  
    if len(current_test_filter)>=min_sentence_len:
        updated_test_data.append(current_test_filter)         
        updated_test_logging.append(test_logging[j])
        updated_test_path.append(test_path[j])
        updated_test_method.append(test_method[j])

under10 =0
over10under200 = 0
over200under400 = 0
over400 = 0

s_sum = 0

for j, curr_row in enumerate(updated_train_data):
    sentence_len_count = 0
    for item in curr_row:
        sentence_len_count += 1
        if sentence_len_count == max_sentence_len-1:
            break
    s_sum += sentence_len_count
    if sentence_len_count <10:
        under10 += 1
    elif sentence_len_count < 200:
        over10under200 += 1
    elif sentence_len_count < 400:
        over200under400 += 1
    else:
        over400 += 1

avg = s_sum/len(updated_train_data)

print(input_filename)
print('updated_train_data')
print("\nSentence length Average : %d\n"%(avg))

print("Under 10 : %d"%(under10))
print("Over 10, Under 200 : %d"%(over10under200))
print("Over 200, Under 400 : %d"%(over200under400))
print("Over 400 : %d\n"%(over400))

test_one = 0
test_zero = 0
train_zero = 0
train_one = 0
for i, data in enumerate(updated_test_logging):
    if data == 1:
        test_one += 1
    else:
        test_zero += 1
for i, data in enumerate(updated_train_logging):
    if data == 0:
        train_zero += 1
    else:
        train_one += 1
print("\nTest_zero: ",test_zero)
print("Train_zero: ",train_zero)
print("zero: ",test_zero+train_zero)
print("Test_one: ",test_one)
print("Train_one: ",train_one)
print("one: ",test_one+train_one)
print()

limit_one = min(train_one,train_zero)
limit_zero = limit_one

final_train_data = []
final_train_logging = []
final_train_path = []
final_train_method = []
count_zero = 0
count_one = 0
for i, item in enumerate(updated_train_logging):
    if item == 0 and count_zero <= limit_zero:
        final_train_data.append(updated_train_data[i])
        final_train_logging.append(updated_train_logging[i])
        final_train_path.append(updated_train_path[i])
        final_train_method.append(updated_train_method[i])
        count_zero += 1
    elif item == 1 and count_one <= limit_one:
        final_train_data.append(updated_train_data[i])
        final_train_logging.append(updated_train_logging[i])
        final_train_path.append(updated_train_path[i])
        final_train_method.append(updated_train_method[i])
        count_one += 1
    if count_zero == limit_zero and count_one == limit_one:
        break

limit_one = min(test_one,test_zero)
limit_zero = limit_one

final_test_data = []
final_test_logging = []
final_test_path = []
final_test_method = []
count_zero = 0
count_one = 0
for i, item in enumerate(updated_test_logging):
    if item == 0 and count_zero <= limit_zero:
        final_test_data.append(updated_test_data[i])
        final_test_logging.append(updated_test_logging[i])
        final_test_path.append(updated_test_path[i])
        final_test_method.append(updated_test_method[i])
        count_zero += 1
    elif item == 1 and count_one <= limit_one:
        final_test_data.append(updated_test_data[i])
        final_test_logging.append(updated_test_logging[i])
        final_test_path.append(updated_test_path[i])
        final_test_method.append(updated_test_method[i])
        count_one += 1
    if count_zero == limit_zero and count_one == limit_one:
        break

train_data_json = "input_data/train/"+str(test_rate)+"/"+filename+".json"
train_len = len(final_train_data)
with open(train_data_json, 'w', encoding="utf-8-sig") as f_train:
    f_train.write('[\r\n')
    for i, item in enumerate(final_train_data):
        f_train.write('\t{\r\n')
        f_train.write('\t\t\"sentence\" : \"')
        for j, item2 in enumerate(item):
            f_train.write(item2+' ')
        f_train.write("\",\r\n")
        f_train.write("\t\t\"isLogged\" : "+str(final_train_logging[i])+",\r\n")
        f_train.write("\t\t\"path\" : \""+final_train_path[i]+"\",\r\n")
        f_train.write("\t\t\"method\" : \""+final_train_method[i]+"\"\r\n")
        if train_len == i+1:
            f_train.write("\t}\r\n")
        else:
            f_train.write("\t},\r\n")
    f_train.write("]")

test_data_json = "input_data/test/"+str(test_rate)+"/"+filename+".json"
test_len = len(final_test_data)
with open(test_data_json, 'w', encoding="utf-8-sig") as f_test:
    f_test.write('[\r\n')
    for i, item in enumerate(final_test_data):
        f_test.write('\t{\r\n')
        f_test.write('\t\t\"sentence\" : \"')
        for j, item2 in enumerate(item):
            f_test.write(item2+' ')
        f_test.write("\",\r\n")
        f_test.write("\t\t\"isLogged\" : "+str(final_test_logging[i])+",\r\n")
        f_test.write("\t\t\"path\" : \""+final_test_path[i]+"\",\r\n")
        f_test.write("\t\t\"method\" : \""+final_test_method[i]+"\"\r\n")
        if test_len == i+1:
            f_test.write("\t}\r\n")
        else:
            f_test.write("\t},\r\n")
    f_test.write("]")

# Create train and test data for deep learning + softmax
X_train = np.empty(shape=[len(final_train_data), max_sentence_len, embed_size_word2vec], dtype='float32')
#X_train = np.zeros(shape=[len(final_train_data), 1, embed_size_word2vec], dtype='float32')
Y_train = np.empty(shape=[len(final_train_logging),1], dtype='int32')
# 1 - start of sentence, # 2 - end of sentence, # 0 - zero padding. Hence, word indices start with 3 
for j, curr_row in enumerate(final_train_data):
    sequence_cnt = 0         
    for item in curr_row:
        if item in vocabulary:
            X_train[j, sequence_cnt, :] = wordvec_model[item] 
            sequence_cnt = sequence_cnt + 1                
            if sequence_cnt == max_sentence_len-1:
                break                
    for k in range(sequence_cnt, max_sentence_len):
        X_train[j, k, :] = np.zeros((1,embed_size_word2vec))        
    Y_train[j,0] = final_train_logging[j]
'''
for j, curr_row in enumerate(final_train_data):
    for item in curr_row:
        if item in vocabulary:
            X_train[j, 0, :] += wordvec_model[item] 
    Y_train[j,0] = final_train_logging[j]
'''

X_test = np.empty(shape=[len(final_test_data), max_sentence_len, embed_size_word2vec], dtype='float32')
#X_test = np.zeros(shape=[len(final_test_data), 1, embed_size_word2vec], dtype='float32')
Y_test = np.empty(shape=[len(final_test_logging),1], dtype='int32')
# 1 - start of sentence, # 2 - end of sentence, # 0 - zero padding. Hence, word indices start with 3 
#'''
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
#'''
'''
for j, curr_row in enumerate(final_test_data):
    for item in curr_row:
        if item in vocabulary:
            X_test[j, 0, :] += wordvec_model[item] 
    Y_test[j,0] = final_test_logging[j]
'''
        
y_train = np_utils.to_categorical(Y_train, 2)
y_test = np_utils.to_categorical(Y_test, 2)

'''
# Construct the deep learning model
inputs = Input(shape=(max_sentence_len, embed_size_word2vec), dtype='float32')
sequence = Masking(mask_value = 0.0)(inputs)
forwards_1 = LSTM(LSTM_output_size)(sequence)
after_dp_forward_4 = Dropout(0.20)(forwards_1) 
backwards_1 = LSTM(LSTM_output_size, go_backwards=True)(sequence)
after_dp_backward_4 = Dropout(0.20)(backwards_1)         
merged = keras.layers.concatenate([after_dp_forward_4, after_dp_backward_4], axis=-1)
after_dp = Dropout(0.5)(merged)
output = Dense(2, activation='softmax')(after_dp)                
output = Dense(2, activation='softmax')(after_dp_forward_4)                
model = Model(input=inputs, output=output)            
rms = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08)
model.compile(loss='categorical_crossentropy', optimizer=rms, metrics=['accuracy'])    

print("Count model parameter.")
model.count_params()
print("Get a short summary of each layer dimensions and parameters.")
model.summary()
'''

clf = DecisionTreeClassifier()

nsamples, nx, ny = X_train.shape
d2_train_dataset = X_train.reshape((nsamples, nx*ny))
nsamples, nx, ny = X_test.shape
d2_test_dataset = X_test.reshape((nsamples, nx*ny))

clf.fit(d2_train_dataset, Y_train)

y_proba = clf.predict_proba(d2_test_dataset)
predictY = np.argmax(y_proba, axis=1)

corrects = np.nonzero(predictY.reshape((-1,1)) == Y_test)
accuracy = len(corrects[0])/len(X_test)*100

preds = y_proba[:,1]
fpr, tpr, threshold = roc_curve(final_test_logging, preds)
roc_auc = auc(fpr, tpr)

idx = 0
true = 0
test_one = 0
test_zero = 0
choose_one = 0
choose_zero = 0

for i, data in enumerate(final_test_logging):
    if data == 1:
        test_one += 1
    else:
        test_zero += 1
for i, data in enumerate(predictY):
    if data == 1:
        choose_one += 1
    else:
        choose_zero += 1

train_one = 0
train_zero = 0
for i, data in enumerate(final_train_logging):
    if data == 0:
        train_zero += 1
    else:
        train_one += 1

print("\nTest accuracy:", accuracy)
print("\ndata size : ", train_zero+train_one+test_zero+test_one)
print("\nzero : ", train_zero + test_zero)
print("\none : ", train_one + test_one)
print("\ntrain_zero : ", train_zero)
print("\ntrain_one : ", train_one)
print("\ntest_zero : ", test_zero)
print("\ntest_one : ", test_one)
print("\nchoose_zero : ", choose_zero)
print("\nchoose_one : ", choose_one)

f1_score = metrics.f1_score(final_test_logging, predictY)
print("\nF1score : ", f1_score)

print("\nAUC : " + str(roc_auc))

print("\nConfusion Matrix")
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

     
#TP, FP, TN, FN file save.
TP_file_name = "Code_snippet/Logging/"+str(test_rate)+"/TP_"+filename+"_"+str(count)+".txt"
FP_file_name = "Code_snippet/Logging/"+str(test_rate)+"/FP_"+filename+"_"+str(count)+ ".txt"
TN_file_name = "Code_snippet/Logging/"+str(test_rate)+"/TN_"+filename+"_"+str(count)+ ".txt"
FN_file_name = "Code_snippet/Logging/"+str(test_rate)+"/FN_"+filename+"_"+str(count)+ ".txt"

f_TP = open(TP_file_name, 'w')
f_FP = open(FP_file_name, 'w')
f_TN = open(TN_file_name, 'w')
f_FN = open(FN_file_name, 'w')

for i, data in enumerate(final_test_logging):
    if data == 1:
        if predictY[i] == 1:
            f_TP.write(final_test_path[i] + '\r\n')
            f_TP.write(final_test_method[i]+ '\r\n')
            f_TP.write(str(predict[i,0]) + '\t' + str(predict[i,1]) + '\r\n')
        else:
            f_FN.write(final_test_path[i]+ '\r\n')
            f_FN.write(final_test_method[i]+ '\r\n')
            f_FN.write(str(predict[i,0]) + '\t' + str(predict[i,1]) + '\r\n')
    else :
        if predictY[i] == 1:
            f_FP.write(final_test_path[i]+ '\r\n')
            f_FP.write(final_test_method[i]+ '\r\n')
            f_FP.write(str(predict[i,0]) + '\t' + str(predict[i,1]) + '\r\n')
        else:
            f_TN.write(final_test_path[i]+ '\r\n')
            f_TN.write(final_test_method[i]+ '\r\n')
            f_TN.write(str(predict[i,0]) + '\t' + str(predict[i,1]) + '\r\n')

f_TP.close()
f_FP.close()
f_TN.close()
f_FN.close()

#model_name = "Model/Logging/"+str(test_rate)+"/JAL_"+filename+str(count)+".pkl"
#joblib.dump(clf, model_name)
'''
plt.figure(roc_count)
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0,1], [0,1], 'r--')
plt.xlim([0,1])
plt.ylim([0,1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
roc_name = "ROC/real_final/"+str(test_rate) +"/JAL_"+filename+"_ep"+str(ep+1)+"_"+str(count)
plt.savefig(roc_name)
roc_count += 1

model_json = model.to_json()
model_name = "Model/real_final/"+str(test_rate) + "/JAL_"+filename+"_ep"+str(ep+1)+"_"+str(count)+"_model.json"
weight_name = "Model/real_final/"+str(test_rate) + "/JAL_"+filename+"_ep"+str(ep+1)+"_"+str(count)+"_model.h5"
with open(model_name,"w") as json_file :
    json_file.write(model_json)
model.save_weights(weight_name)
print("Saved model to disk\n\n\n")
'''
del clf
