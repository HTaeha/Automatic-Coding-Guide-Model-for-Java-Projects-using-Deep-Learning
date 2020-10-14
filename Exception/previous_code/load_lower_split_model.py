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

test_file = sys.argv[1]
test_type = sys.argv[2]
model_file = sys.argv[3]
model_type = sys.argv[4]

test_filename = test_file + '-' + test_type#"glassfish-CAST_s(partial)"
model_filename = model_file + '-' + model_type#"hbase-CAST_s"

test_java_auto_exception_json = '../MakeJSON/output/Exception/final/'+test_filename+'.json'

model_ep = 15
model_test_rate = int(sys.argv[5])
model_count = 1

filename = "load_"+model_filename+"-test_"+test_filename
count = 1

#If balance = False, test all data
#If balance = True, test_limit_zero + test_limit_one
balance = sys.argv[6]
test_limit_zero = int(sys.argv[7])
test_limit_one = int(sys.argv[8])

#========================================================================================
# Initializing Hyper parameter
#========================================================================================
#1. Classifier hyperparameters
numCV = 1
min_sentence_len = 10
max_sentence_len = 400
batch_size = 32

embed_size_word2vec = 200

# Load the word2vec model and vocabulary
wordvec_path = "Wordvec_Model/final/" + str(model_test_rate) + "/" + model_filename +"_" + str(model_count) + ".model"
wordvec_model = Word2Vec.load(wordvec_path)
vocabulary = wordvec_model.wv.vocab

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
#current_data = current_sentence_filter
    current_data = current_sentence_tokens
    current_data = list(filter(None, current_data))
    lowercase_data = []
    for word in current_data:
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
            lowercase_data.append(data.lower())
    test_data.append(lowercase_data)
    test_exception.append(item['isException'])
    test_path.append(item['path'])
    test_method.append(item['method'])



#========================================================================================
# Split cross validation sets and perform deep learning + softmax based classification
#========================================================================================
totalLength = len(test_data)
splitLength = int(totalLength / 10)

for old_index in range(len(test_data)):
	new_index = np.random.randint(len(test_data))
	test_data[old_index], test_data[new_index] = test_data[new_index], test_data[old_index]
	test_exception[old_index], test_exception[new_index] = test_exception[new_index], test_exception[old_index]
	test_path[old_index], test_path[new_index] = test_path[new_index], test_path[old_index]
	test_method[old_index], test_method[new_index] = test_method[new_index], test_method[old_index]

if balance == True:
    test_data = test_data[(model_test_rate-1)*splitLength:model_test_rate*splitLength-1]
    test_exception = test_exception[(model_test_rate-1)*splitLength:model_test_rate*splitLength-1]
    test_path = test_path[(model_test_rate-1)*splitLength:model_test_rate*splitLength-1]
    test_method = test_method[(model_test_rate-1)*splitLength:model_test_rate*splitLength-1]

# Remove words outside the vocabulary
updated_test_data = []
updated_test_exception = []
updated_test_path = []
updated_test_method = []

for j, item in enumerate(test_data):
    current_test_filter = [word for word in item if word in vocabulary]  
    if len(current_test_filter)>=min_sentence_len:
        updated_test_data.append(current_test_filter)         
        updated_test_exception.append(test_exception[j])
        updated_test_path.append(test_path[j])
        updated_test_method.append(test_method[j])

if balance == True:
    final_test_data = []
    final_test_exception = []
    final_test_path = []
    final_test_method = []
    count_zero = 0
    count_one = 0
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
else:
    final_test_data = updated_test_data
    final_test_exception = updated_test_exception
    final_test_path = updated_test_path
    final_test_method = updated_test_method

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
        
# Load model
model_json = "Model/final/"+str(model_test_rate)+"/JAE_"+model_filename+"_"+str(model_count)+"_model.json"
model_h5 = "Model/final/"+str(model_test_rate)+"/JAE_"+ model_filename+"_"+str(model_count)+"_model.h5"
json_file = open(model_json, "r")
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights(model_h5)
print("Loaded model from disk")

rms = RMSprop(lr = 0.001, rho = 0.9, epsilon = 1e-08)
model.compile(loss = 'categorical_crossentropy', optimizer = rms, metrics = ['accuracy'])
predict = model.predict(X_test)

preds = predict[:,1]
fpr, tpr, threshold = roc_curve(final_test_exception, preds)
roc_auc = auc(fpr, tpr)

predictY=[]
for k in predict:
    predictY.append(list(k).index(max(k)))

idx = 0
true = 0
test_one = 0
test_zero = 0
choose_one = 0
choose_zero = 0

#TP, FP, TN, FN file save.
TP_file_name = "Code_snippet/final/"+str(model_test_rate)+"/TP_"+filename+"_"+str(count)+".txt"
FP_file_name = "Code_snippet/final/"+str(model_test_rate)+"/FP_"+filename+"_"+str(count)+ ".txt"
TN_file_name = "Code_snippet/final/"+str(model_test_rate)+"/TN_"+filename+"_"+str(count)+ ".txt"
FN_file_name = "Code_snippet/final/"+str(model_test_rate)+"/FN_"+filename+"_"+str(count)+ ".txt"

f_TP = open(TP_file_name, 'w')
f_FP = open(FP_file_name, 'w')
f_TN = open(TN_file_name, 'w')
f_FN = open(FN_file_name, 'w')

for i, data in enumerate(final_test_exception):
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


#ROC graph
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0,1], [0,1], 'r--')
plt.xlim([0,1])
plt.ylim([0,1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
roc_name = "ROC/final/"+str(model_test_rate)+"/JAE_"+filename+"_"+str(count)
plt.savefig(roc_name)
