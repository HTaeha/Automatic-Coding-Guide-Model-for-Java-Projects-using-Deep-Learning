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

test_filename = "hbase-function_AST"
test_filename2 = "hbase-original_code"
code_snippet_filename = "hbase-CAST_d"
code_snippet_filename2 = "hbase-code"

load_model_filename = "doc-hbase-function_AST-hbase-original_code_balanced_max400_masking"
model_code_filename = test_filename2 + "_balanced_max400_masking"

test_java_auto_exception_json = '../MakeJSON/output/Exception/'+test_filename+'.json'
test_java_auto_exception_json2 = '../MakeJSON/output/Exception/'+test_filename2+'.json'

model_ep = 15
#1, 10, cross-project
model_test_rate = '10'
model_count = 1

filename = "analysis-hbase_CAST_d_code"
count = 1

#If balance = False, test all data
#If balance = True, test_limit_zero + test_limit_one
balance = True
test_limit_zero = 1125
test_limit_one = 1125
test_limit_zero_code = 1707
test_limit_one_code = 1707
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

embed_size_word2vec = 200

# Load the word2vec model and vocabulary
wordvec_path = "Wordvec_Model/" + str(model_test_rate) + "/" + load_model_filename + "_min" + str(min_sentence_len) + "_" + str(model_count) + ".model"
wordvec_model = Word2Vec.load(wordvec_path)
vocabulary = wordvec_model.wv.vocab

wordvec_path_code = "Wordvec_Model/" + str(model_test_rate) + "/" + model_code_filename + "_min" + str(min_sentence_len2) + "_" + str(model_count) + ".model"
wordvec_model_code = Word2Vec.load(wordvec_path_code)
vocabulary_code = wordvec_model_code.wv.vocab

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

#Preprocessing test data2
with open(test_java_auto_exception_json2,encoding='utf-8-sig') as test_data_file2:
	t_data2 = json.loads(test_data_file2.read(), strict=False)
    
test_data2 = []
test_exception2 = []
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
    test_exception2.append(item['isException'])
    test_path2.append(item['path'])
    test_method2.append(item['method'])



#========================================================================================
# Split cross validation sets and perform deep learning + softmax based classification
#========================================================================================
totalLength = len(test_data)
splitLength = int(totalLength / 10)

totalLength_code = len(test_data2)
splitLength_code = int(totalLength_code / 10)

for old_index in range(len(test_data)):
	new_index = np.random.randint(old_index+1)
	test_data[old_index], test_data[new_index] = test_data[new_index], test_data[old_index]
	test_exception[old_index], test_exception[new_index] = test_exception[new_index], test_exception[old_index]
	test_path[old_index], test_path[new_index] = test_path[new_index], test_path[old_index]
	test_method[old_index], test_method[new_index] = test_method[new_index], test_method[old_index]
	test_data2[old_index], test_data2[new_index] = test_data2[new_index], test_data2[old_index]
	test_exception2[old_index], test_exception2[new_index] = test_exception2[new_index], test_exception2[old_index]
	test_path2[old_index], test_path2[new_index] = test_path2[new_index], test_path2[old_index]
	test_method2[old_index], test_method2[new_index] = test_method2[new_index], test_method2[old_index]

test_data_code = test_data2
test_exception_code = test_exception2
test_path_code = test_path2
test_method_code = test_method2

if model_test_rate == 1:
	test_data = test_data[:splitLength-1]
	test_exception = test_exception[:splitLength-1]
	test_path = test_path[:splitLength-1]
	test_method = test_method[:splitLength-1]
	
	test_data2 = test_data2[:splitLength-1]
	test_exception2 = test_exception2[:splitLength-1]
	test_path2 = test_path2[:splitLength-1]
	test_method2 = test_method2[:splitLength-1]
	
	test_data_code = test_data_code[:splitLength_code-1]
	test_exception_code = test_exception_code[:splitLength_code-1]
	test_path_code = test_path_code[:splitLength_code-1]
	test_method_code = test_method_code[:splitLength_code-1]

elif model_test_rate == 10:
	test_data = test_data[9*splitLength:]
	test_exception = test_exception[9*splitLength:]
	test_path = test_path[9*splitLength:]
	test_method = test_method[9*splitLength:]
	
	test_data2 = test_data2[9*splitLength:]
	test_exception2 = test_exception2[9*splitLength:]
	test_path2 = test_path2[9*splitLength:]
	test_method2 = test_method2[9*splitLength:]

	test_data_code = test_data_code[9*splitLength_code:]
	test_exception_code = test_exception_code[9*splitLength_code:]
	test_path_code = test_path_code[9*splitLength_code:]
	test_method_code = test_method_code[9*splitLength_code:]

# Remove words outside the vocabulary
updated_test_data = []
updated_test_exception = []
updated_test_path = []
updated_test_method = []

updated_test_data2 = []
updated_test_exception2 = []
updated_test_path2 = []
updated_test_method2 = []

updated_test_data_code = []
updated_test_exception_code = []
updated_test_path_code = []
updated_test_method_code = []

for j, item in enumerate(test_data):
	current_test_filter = [word for word in item if word in vocabulary]  
	current_test_filter2 = [word2 for word2 in test_data2[j] if word2 in vocabulary]  
	if len(current_test_filter)>=min_sentence_len and len(current_test_filter2)>=min_sentence_len2:
		updated_test_data.append(current_test_filter)         
		updated_test_exception.append(test_exception[j])
		updated_test_path.append(test_path[j])
		updated_test_method.append(test_method[j])

		updated_test_data2.append(current_test_filter2)
		updated_test_exception2.append(test_exception2[j])
		updated_test_path2.append(test_path2[j])
		updated_test_method2.append(test_method2[j])

for j, item in enumerate(test_data2):
	current_test_filter_code = [word for word in item if word in vocabulary_code]
	if len(current_test_filter_code) >= min_sentence_len2:
		updated_test_data_code.append(current_test_filter_code)
		updated_test_exception_code.append(test_exception_code[j])
		updated_test_path_code.append(test_path_code[j])
		updated_test_method_code.append(test_method_code[j])

if balance == True:
	final_test_data = []
	final_test_exception = []
	final_test_path = []
	final_test_method = []

	final_test_data2 = []
	final_test_exception2 = []
	final_test_path2 = []
	final_test_method2 = []

	final_test_data_code = []
	final_test_exception_code = []
	final_test_path_code = []
	final_test_method_code = []


	count_zero = 0
	count_one = 0
	for i, item in enumerate(updated_test_exception):
		if item == 0 and count_zero <= test_limit_zero:
			final_test_data.append(updated_test_data[i])
			final_test_exception.append(updated_test_exception[i])
			final_test_path.append(updated_test_path[i])
			final_test_method.append(updated_test_method[i])

			final_test_data2.append(updated_test_data2[i])
			final_test_exception2.append(updated_test_exception2[i])
			final_test_path2.append(updated_test_path2[i])
			final_test_method2.append(updated_test_method2[i])
            
			count_zero += 1
		elif item == 1 and count_one <= test_limit_one:
			final_test_data.append(updated_test_data[i])
			final_test_exception.append(updated_test_exception[i])
			final_test_path.append(updated_test_path[i])
			final_test_method.append(updated_test_method[i])

			final_test_data2.append(updated_test_data2[i])
			final_test_exception2.append(updated_test_exception2[i])
			final_test_path2.append(updated_test_path2[i])
			final_test_method2.append(updated_test_method2[i])
            
			count_one += 1
		if count_zero == test_limit_zero and count_one == test_limit_one:
			break

	count_zero = 0
	count_one = 0
	for i, item in enumerate(updated_test_exception_code):
		if item == 0 and count_zero <= test_limit_zero_code:
			final_test_data_code.append(updated_test_data_code[i])
			final_test_exception_code.append(updated_test_exception[i])
			final_test_path_code.append(updated_test_path_code[i])
			final_test_method_code.append(updated_test_method_code[i])
			count_zero += 1
		elif item == 1 and count_zero <= test_limit_one_code:
			final_test_data_code.append(updated_test_data_code[i])
			final_test_exception_code.append(updated_test_exception[i])
			final_test_path_code.append(updated_test_path_code[i])
			final_test_method_code.append(updated_test_method_code[i])
			count_one += 1
		if count_zero == test_limit_zero_code and count_one == test_limit_one_code:
			break
else:
	final_test_data = updated_test_data
	final_test_exception = updated_test_exception
	final_test_path = updated_test_path
	final_test_method = updated_test_method

	final_test_data2 = updated_test_data2
	final_test_exception2 = updated_test_exception2
	final_test_path2 = updated_test_path2
	final_test_method2 = updated_test_method2

doc_X_test = np.empty(shape=[len(final_test_data), max_sentence_len + max_sentence_len2, embed_size_word2vec], dtype='float32')
Y_test = np.empty(shape=[len(final_test_exception),1], dtype='int32')
# 1 - start of sentence, # 2 - end of sentence, # 0 - zero padding. Hence, word indices start with 3 
for j, curr_row in enumerate(final_test_data):
	sequence_cnt = 0          
	for item in curr_row:
		if item in vocabulary:
			doc_X_test[j, sequence_cnt, :] = wordvec_model[item]
			sequence_cnt = sequence_cnt + 1                
			if sequence_cnt == max_sentence_len-1:
				break                
	for k in range(sequence_cnt, max_sentence_len):
		doc_X_test[j, k, :] = np.zeros((1,embed_size_word2vec))
	Y_test[j,0] = final_test_exception[j]
        
# 1 - start of sentence, # 2 - end of sentence, # 0 - zero padding. Hence, word indices start with 3 
for j, curr_row in enumerate(final_test_data2):
	sequence_cnt = 0          
	for item in curr_row:
		if item in vocabulary:
			doc_X_test[j, max_sentence_len + sequence_cnt, :] = wordvec_model[item]
			sequence_cnt = sequence_cnt + 1                
			if sequence_cnt == max_sentence_len2-1:
				break                
	for k in range(sequence_cnt, max_sentence_len2):
		doc_X_test[j, max_sentence_len + k, :] = np.zeros((1,embed_size_word2vec))

X_test_code = np.empty(shape=[len(final_test_data_code), max_sentence_len2, embed_size_word2vec], dtype='float32')
Y_test_code = np.empty(shape=[len(final_test_exception_code),1], dtype='int32')
for j, curr_row in enumerate(final_test_data_code):
	sequence_cnt = 0
	for item in curr_row:
		if item in vocabulary_code:
			X_test_code[j, sequence_cnt, :] = wordvec_model_code[item]
			sequence_cnt += 1
			if sequence_cnt == max_sentence_len2-1:
				break
	for k in range(sequence_cnt, max_sentence_len2):
		X_test_code[j, k, :] = np.zeros((1,embed_size_word2vec))
	Y_test_code[j,0] = final_test_exception_code[j]

# Load model
model_json = "Model/"+str(model_test_rate)+"/JAL_"+ load_model_filename+"_ep"+str(model_ep)+"_min"+str(min_sentence_len)+"_"+str(model_count)+"_model.json"
model_h5 = "Model/"+str(model_test_rate)+"/JAL_"+ load_model_filename+"_ep"+str(model_ep)+"_min"+str(min_sentence_len)+"_"+str(model_count)+"_model.h5"
json_file = open(model_json, "r")
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights(model_h5)

model_json_code = "Model/"+str(model_test_rate)+"/JAL_"+model_code_filename+"_ep"+str(model_ep)+"_min"+str(min_sentence_len2)+"_"+str(model_count)+"_model.json"
model_h5_code = "Model/"+str(model_test_rate)+"/JAL_"+model_code_filename+"_ep"+str(model_ep)+"_min"+str(min_sentence_len2)+"_"+str(model_count)+"_model.h5"
json_file_code = open(model_json_code, "r")
loaded_model_json_code = json_file_code.read()
json_file_code.close()
model_code = model_from_json(loaded_model_json_code)
model_code.load_weights(model_h5_code)
print("Loaded model from disk")

rms = RMSprop(lr = 0.001, rho = 0.9, epsilon = 1e-08)
model.compile(loss = 'categorical_crossentropy', optimizer = rms, metrics = ['accuracy'])
model_code.compile(loss = 'categorical_crossentropy', optimizer = rms, metrics = ['accuracy'])
predict = model.predict(doc_X_test)
predict_code = model_code.predict(X_test_code)

preds = predict[:,1]
preds_code = predict_code[:,1]
fpr, tpr, threshold = roc_curve(final_test_exception, preds)
fpr_code, tpr_code, threshold_code = roc_curve(final_test_exception_code, preds_code)
roc_auc = auc(fpr, tpr)
roc_auc_code = auc(fpr_code, tpr_code)

predictY=[]
for k in predict:
    predictY.append(list(k).index(max(k)))

predictY_code = []
for k in predict_code:
	predictY_code.append(list(k).index(max(k)))

print(code_snippet_filename)
print(code_snippet_filename2)
print()

#ASTT_codeT, ASTT_codeF, ASTF_codeT, ASTF_codeF file save.
ASTT_codeT_file_name = "Analysis/Code_snippet/"+str(model_test_rate)+"/ASTT_codeT_"+filename+"_min" + str(min_sentence_len) + "_ep" + str(model_ep) +"_"+str(count)+".txt"
ASTT_codeF_file_name = "Analysis/Code_snippet/"+str(model_test_rate)+"/ASTT_codeF_"+filename+"_min" + str(min_sentence_len) + "_ep" + str(model_ep) +"_"+str(count)+ ".txt"
ASTF_codeT_file_name = "Analysis/Code_snippet/"+str(model_test_rate)+"/ASTF_codeT_"+filename+"_min" + str(min_sentence_len) + "_ep" + str(model_ep) +"_"+str(count)+ ".txt"
ASTF_codeF_file_name = "Analysis/Code_snippet/"+str(model_test_rate)+"/ASTF_codeF_"+filename+"_min" + str(min_sentence_len) + "_ep" + str(model_ep) +"_"+str(count)+ ".txt"

f_ASTT_codeT = open(ASTT_codeT_file_name, 'w')
f_ASTT_codeF = open(ASTT_codeF_file_name, 'w')
f_ASTF_codeT = open(ASTF_codeT_file_name, 'w')
f_ASTF_codeF = open(ASTF_codeF_file_name, 'w')

AST1_code1 = 0
AST1_code0 = 0
AST0_code1 = 0
AST0_code0 = 0

ASTT_codeT = 0
ASTT_codeF = 0
ASTF_codeT = 0
ASTF_codeF = 0
for i, data in enumerate(final_test_exception):
	if predictY[i] == 1:
		for i2, data2 in enumerate(final_test_exception_code):
			if final_test_path[i] == final_test_path_code[i2] and final_test_method[i] == final_test_method_code[i2]:
				if predictY_code[i2] == 1:
					AST1_code1 += 1
				else:
					AST1_code0 += 1
	else:
		for i2, data2 in enumerate(final_test_exception_code):
			if final_test_path[i] == final_test_path_code[i2] and final_test_method[i] == final_test_method_code[i2]:
				if predictY_code[i2] == 1:
					AST0_code1 += 1
				else:
					AST0_code0 += 1
	if predictY[i] == Y_test[i][0]:
		for i2, data2 in enumerate(final_test_exception_code):
			if final_test_path[i] == final_test_path_code[i2] and final_test_method[i] == final_test_method_code[i2]:
				if predictY_code[i2] == Y_test_code[i2][0]:
					ASTT_codeT += 1
					f_ASTT_codeT.write(final_test_path[i] + '\r\n')
					f_ASTT_codeT.write(final_test_method[i]+ '\r\n')
					f_ASTT_codeT.write(code_snippet_filename+'\t'+str(predict[i,0]) + '\t' + str(predict[i,1]) + '\r\n')
					f_ASTT_codeT.write("predict: "+str(predictY[i])+"\tanswer: "+str(Y_test[i][0])+'\r\n')
					f_ASTT_codeT.write(code_snippet_filename2+'\t'+str(predict_code[i,0]) + '\t' + str(predict_code[i,1]) + '\r\n')
					f_ASTT_codeT.write("predict: "+str(predictY_code[i])+"\tanswer: "+str(Y_test_code[i][0])+'\r\n')
				else:
					ASTT_codeF += 1
					f_ASTT_codeF.write(final_test_path[i]+ '\r\n')
					f_ASTT_codeF.write(final_test_method[i]+ '\r\n')
					f_ASTT_codeF.write(code_snippet_filename+'\t'+str(predict[i,0]) + '\t' + str(predict[i,1]) + '\r\n')
					f_ASTT_codeF.write("predict: "+str(predictY[i])+"\tanswer: "+str(Y_test[i][0])+'\r\n')
					f_ASTT_codeF.write(code_snippet_filename2+'\t'+str(predict_code[i,0]) + '\t' + str(predict_code[i,1]) + '\r\n')
					f_ASTT_codeF.write("predict: "+str(predictY_code[i])+"\tanswer: "+str(Y_test_code[i][0])+'\r\n')
	else:
		for i2, data2 in enumerate(final_test_exception_code):
			if final_test_path[i] == final_test_path_code[i2] and final_test_method[i] == final_test_method_code[i2]:
				if predictY_code[i2] == Y_test_code[i2][0]:
					ASTF_codeT += 1
					f_ASTF_codeT.write(final_test_path[i]+ '\r\n')
					f_ASTF_codeT.write(final_test_method[i]+ '\r\n')
					f_ASTF_codeT.write(code_snippet_filename+'\t'+str(predict[i,0]) + '\t' + str(predict[i,1]) + '\r\n')
					f_ASTF_codeT.write("predict: "+str(predictY[i])+"\tanswer: "+str(Y_test[i][0])+'\r\n')
					f_ASTF_codeT.write(code_snippet_filename2+'\t'+str(predict_code[i,0]) + '\t' + str(predict_code[i,1]) + '\r\n')
					f_ASTF_codeT.write("predict: "+str(predictY_code[i])+"\tanswer: "+str(Y_test_code[i][0])+'\r\n')
				else:
					ASTF_codeF += 1
					f_ASTF_codeF.write(final_test_path[i]+ '\r\n')
					f_ASTF_codeF.write(final_test_method[i]+ '\r\n')
					f_ASTF_codeF.write(code_snippet_filename+'\t'+str(predict[i,0]) + '\t' + str(predict[i,1]) + '\r\n')
					f_ASTF_codeF.write("predict: "+str(predictY[i])+"\tanswer: "+str(Y_test[i][0])+'\r\n')
					f_ASTF_codeF.write(code_snippet_filename2+'\t'+str(predict_code[i,0]) + '\t' + str(predict_code[i,1]) + '\r\n')
					f_ASTF_codeF.write("predict: "+str(predictY_code[i])+"\tanswer: "+str(Y_test_code[i][0])+'\r\n')

f_ASTT_codeT.close()
f_ASTT_codeF.close()
f_ASTF_codeT.close()
f_ASTF_codeF.close()


				
print("AST1_code1 : "+str(AST1_code1))
print("AST1_code0 : "+str(AST1_code0))
print("AST0_code1 : "+str(AST0_code1))
print("AST0_code0 : "+str(AST0_code0))
print()
print("ASTT_codeT : ", ASTT_codeT)
print("ASTT_codeF : ", ASTT_codeF)
print("ASTF_codeT : ", ASTF_codeT)
print("ASTF_codeF : ", ASTF_codeF)

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
for i, data in enumerate(final_test_exception_code):
	if data == 1:
		test_one += 1
	else:
		test_zero += 1
for i, data in enumerate(predictY_code):
	if data == 1:
		choose_one = choose_one + 1
	if data == Y_test_code[idx][0]:
		true = true +1
	idx = idx + 1
for i, data in enumerate(predictY_code):
	if data == 0:
		choose_zero = choose_zero + 1
accuracy = (float(true)/len(predict_code))*100

print("\nTest accuracy:", accuracy)
print("\ntest_zero : ", test_zero)
print("\ntest_one : ", test_one)
print("\nchoose_zero : ", choose_zero)
print("\nchoose_one : ", choose_one)


f1_score = metrics.f1_score(final_test_exception_code, predictY_code)
print("\nF1score : ", f1_score)

print("\nAUC : " + str(roc_auc_code))


print("\nConfusion Matrix2")
print(confusion_matrix(Y_test_code, predictY_code))
cm2 = confusion_matrix(Y_test_code, predictY_code)
cm2 = cm2.astype('float') / cm2.sum(axis=1)[:, np.newaxis]
for p in range(cm2.shape[0]):
	print('True label', p)
	for q in range(cm2.shape[0]):
		print(cm2[p,q], end=' ')
		if q%100 == 0:
			print(' ')
	print(' ')

plt.figure(1)
plt.title('hbase-CAST_d')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.4f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0,1], [0,1], 'r--')
plt.xlim([0,1])
plt.ylim([0,1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
roc_name = "Analysis/ROC/"+str(model_test_rate)+"/"+code_snippet_filename+"_balanced_max400_masking_ep"+str(model_ep)+"_min"+str(min_sentence_len)+"_"+str(model_count)
plt.savefig(roc_name)

