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

train_filename = 'hbase-original_code'
test_filename = 'hbase-function'

java_auto_exception_json = '../MakeJSON/output/Exception/'+train_filename+'.json'
test_java_auto_exception_json = '../MakeJSON/output/Exception/'+test_filename+'.json'
filename = 'train_'+train_filename+'-test_'+test_filename

epoch_len = 15
count = 1

data_size_check = False

train_limit_zero = 9000
train_limit_one = 8500
test_limit_zero = 1000
test_limit_one = 1000

#========================================================================================
# Initializing Hyper parameter
#========================================================================================
#1. Word2vec parameters
min_word_frequency_word2vec = 5
embed_size_word2vec = 200
context_window_word2vec = 5

#2. Classifier hyperparameters
numCV = 1
min_sentence_len = 10
max_sentence_len = 200
batch_size = 32

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

# Learn the word2vec model and extract vocabulary
wordvec_model = Word2Vec(all_data, min_count=min_word_frequency_word2vec, size=embed_size_word2vec, window=context_window_word2vec)
vocabulary = wordvec_model.wv.vocab
if data_size_check == False:
    wordvec_model.save("Wordvec_Model/cross-project/" + filename + "_min" + str(min_sentence_len) + "_" + str(count) + ".model")

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
    test_data.append(current_data)
    test_exception.append(item['isException'])
    test_path.append(item['path'])
    test_method.append(item['method'])



#========================================================================================
# Split cross validation sets and perform deep learning + softmax based classification
#========================================================================================
for old_index in range(len(all_data)):
    new_index = np.random.randint(old_index+1)
    all_data[old_index], all_data[new_index] = all_data[new_index], all_data[old_index]
    all_exception[old_index], all_exception[new_index] = all_exception[new_index], all_exception[old_index]
    all_path[old_index], all_path[new_index] = all_path[new_index], all_path[old_index]
    all_method[old_index], all_method[new_index] = all_method[new_index], all_method[old_index]

# Split cross validation set
train_data = all_data
train_exception = all_exception
train_path = all_path
train_method = all_method

updated_train_data = []    
updated_train_exception = []
updated_train_path = []
updated_train_method = []

for j, item in enumerate(train_data):
    current_train_filter = [word for word in item if word in vocabulary]
    if len(current_train_filter)>=min_sentence_len:
        updated_train_data.append(current_train_filter)
        updated_train_exception.append(train_exception[j])  
        updated_train_path.append(train_path[j])
        updated_train_method.append(train_method[j])

final_train_data = []
final_train_exception = []
final_train_path = []
final_train_method = []
count_zero = 0
count_one = 0
for i, item in enumerate(updated_train_exception):
    if item == 0 and count_zero <= train_limit_zero:
        final_train_data.append(updated_train_data[i])
        final_train_exception.append(updated_train_exception[i])
        final_train_path.append(updated_train_path[i])
        final_train_method.append(updated_train_method[i])
        count_zero += 1
    elif item == 1 and count_one <= train_limit_one:
        final_train_data.append(updated_train_data[i])
        final_train_exception.append(updated_train_exception[i])
        final_train_path.append(updated_train_path[i])
        final_train_method.append(updated_train_method[i])
        count_one += 1
    if count_zero == train_limit_zero and count_one == train_limit_one:
        break

# Create train and test data for deep learning + softmax
X_train = np.empty(shape=[len(final_train_data), max_sentence_len, embed_size_word2vec], dtype='float32')
Y_train = np.empty(shape=[len(final_train_exception),1], dtype='int32')
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
    Y_train[j,0] = final_train_exception[j]





for old_index in range(len(test_data)):
	new_index = np.random.randint(old_index+1)
	test_data[old_index], test_data[new_index] = test_data[new_index], test_data[old_index]
	test_exception[old_index], test_exception[new_index] = test_exception[new_index], test_exception[old_index]
	test_path[old_index], test_path[new_index] = test_path[new_index], test_path[old_index]
	test_method[old_index], test_method[new_index] = test_method[new_index], test_method[old_index]

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

if data_size_check == True:
    test_one = 0
    test_zero = 0
    train_zero = 0
    train_one = 0
    for i, data in enumerate(updated_test_exception):
        if data == 1:
            test_one += 1
        else:
            test_zero += 1
    for i, data in enumerate(updated_train_exception):
        if data == 0:
            train_zero += 1
        else:
            train_one += 1
    print("Test_zero: ", test_zero)
    print("Train_zero: ", train_zero)
    print("zero: ", test_zero + train_zero)
    print("Test_one: ", test_one)
    print("Train_one: ", train_one)
    print("one: ", test_one+train_one)
    sys.exit(1)


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
        
y_train = np_utils.to_categorical(Y_train, 2)
y_test = np_utils.to_categorical(Y_test, 2)

# Construct the deep learning model
sequence = Input(shape=(max_sentence_len, embed_size_word2vec), dtype='float32')
forwards_1 = LSTM(1024)(sequence)
after_dp_forward_4 = Dropout(0.20)(forwards_1) 
backwards_1 = LSTM(1024, go_backwards=True)(sequence)
after_dp_backward_4 = Dropout(0.20)(backwards_1)         
merged = keras.layers.concatenate([after_dp_forward_4, after_dp_backward_4], axis=-1)
after_dp = Dropout(0.5)(merged)
output = Dense(2, activation='softmax')(after_dp)                
model = Model(input=sequence, output=output)            
rms = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08)
model.compile(loss='categorical_crossentropy', optimizer=rms, metrics=['accuracy'])    

roc_count = 1
for ep in range(epoch_len) :
    print(str(ep+1) + "\n")
    hist = model.fit(X_train, y_train, batch_size=batch_size, epochs=1, verbose=2)

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

#	'''
#TP, FP, TN, FN file save.
    TP_file_name = "Code_snippet/cross-project/TP_"+filename+"_min" + str(min_sentence_len) + "_ep" + str(ep+1) +"_"+str(count)+".txt"
    FP_file_name = "Code_snippet/cross-project/FP_"+filename+"_min" + str(min_sentence_len) + "_ep" + str(ep+1) +"_"+str(count)+ ".txt"
    TN_file_name = "Code_snippet/cross-project/TN_"+filename+"_min" + str(min_sentence_len) + "_ep" + str(ep+1) +"_"+str(count)+ ".txt"
    FN_file_name = "Code_snippet/cross-project/FN_"+filename+"_min" + str(min_sentence_len) + "_ep" + str(ep+1) +"_"+str(count)+ ".txt"

    f_TP = open(TP_file_name, 'w')
    f_FP = open(FP_file_name, 'w')
    f_TN = open(TN_file_name, 'w')
    f_FN = open(FN_file_name, 'w')

#	print("\nreal test exception data")
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

#'''
    for i, data in enumerate(final_test_exception):
        if data == 1:
            test_one += 1
        else:
            test_zero += 1
#	print("\npredict test exception data")
    for i, data in enumerate(predictY):
        if data == 1:
            choose_one = choose_one + 1
#			print(choose_one)
#			print(final_test_data[i])
#			print(final_test_path[i])
        if data == Y_test[idx][0]:
            true = true +1
        idx = idx + 1
#	print("\npredict test not exception data")
    for i, data in enumerate(predictY):
        if data == 0:
            choose_zero = choose_zero + 1
#			print(choose_zero)
#			print(final_test_data[i])
#			print(final_test_path[i])
    accuracy = (float(true)/len(predict))*100

    train_one = 0
    train_zero = 0
    for i, data in enumerate(final_train_exception):
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
	

    train_result = hist.history
    print("\nTrain_result", train_result)

#ROC graph
	plt.figure(roc_count)
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0,1], [0,1], 'r--')
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    roc_name = "ROC/cross-project/JAL_"+filename+"_ep"+str(ep+1)+"_min"+str(min_sentence_len)+"_"+str(count)
    plt.savefig(roc_name)
	roc_count += 1

#Save model
    model_json = model.to_json()
    model_name = "Model/cross-project/JAL_"+filename+"_ep"+str(ep+1)+"_min"+str(min_sentence_len)+"_"+str(count)+"_model.json"
    weight_name = "Model/cross-project/JAL_"+filename+"_ep"+str(ep+1)+"_min"+str(min_sentence_len)+"_"+str(count)+"_model.h5"
    with open(model_name,"w") as json_file :
        json_file.write(model_json)
    model.save_weights(weight_name)
    print("Saved model to disk\n\n\n")
del model
