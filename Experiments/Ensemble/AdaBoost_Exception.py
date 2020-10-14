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
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.ensemble import AdaBoostRegressor
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

input_file = sys.argv[1]
input_type1 = sys.argv[2]
filename = input_file + '-AdaBoost_'+input_type1
input_filename1 = input_file + '-' + input_type1
model_filename = input_filename1
java_auto_exception_json = '../MakeJSON/output/Exception/real_final/'+input_filename1+'.json'
wordvec_model_name1 = input_filename1
epoch_len = 15
model_test_rate = int(sys.argv[3])
count = 1
model_count = 1

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
max_sentence_len1 = 400
batch_size = 32
LSTM_output_size = 256

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
print("First data")
print("zero : ")
print(zero)
print("one : ")
print(one)
print()

# Learn the word2vec model and extract vocabulary
'''
wordvec_model = Word2Vec(all_data, min_count=min_word_frequency_word2vec, size=embed_size_word2vec, window=context_window_word2vec)
vocabulary = wordvec_model.wv.vocab
wordvec_model.save("Wordvec_Model/Exception/"+wordvec_model_name1+"_"+str(count)+".model")

sys.exit(1)
'''
wordvec_path = "../JavaAutoException/Wordvec_Model/real_final/"+wordvec_model_name1+"_"+str(model_count)+".model"
wordvec_model = Word2Vec.load(wordvec_path)
vocabulary = wordvec_model.wv.vocab

#========================================================================================
# Split cross validation sets and perform deep learning + softmax based classification
#========================================================================================
totalLength = len(all_data)
splitLength = int(totalLength / 10)
#'''
for old_index in range(len(all_data)):
    new_index = np.random.randint(len(all_data))
    all_data[old_index], all_data[new_index] = all_data[new_index], all_data[old_index]
    all_exception[old_index], all_exception[new_index] = all_exception[new_index], all_exception[old_index]
    all_path[old_index], all_path[new_index] = all_path[new_index], all_path[old_index]
    all_method[old_index], all_method[new_index] = all_method[new_index], all_method[old_index]

#'''
# Split cross validation set

if model_test_rate == 1:
    train_data = all_data[splitLength:]
    test_data = all_data[:splitLength-1]
    train_exception = all_exception[splitLength:]
    test_exception = all_exception[:splitLength-1]
    train_path = all_path[splitLength:]
    test_path = all_path[:splitLength]
    train_method = all_method[splitLength:]
    test_method = all_method[:splitLength]
else: 
    train_data = all_data[:(model_test_rate-1)*splitLength-1] + all_data[model_test_rate*splitLength:]
    test_data = all_data[(model_test_rate-1)*splitLength:model_test_rate*splitLength-1]
    train_exception = all_exception[:(model_test_rate-1)*splitLength-1]+ all_exception[model_test_rate*splitLength:]
    test_exception = all_exception[(model_test_rate-1)*splitLength:model_test_rate*splitLength-1]
    train_path = all_path[:(model_test_rate-1)*splitLength-1]+ all_path[model_test_rate*splitLength:]
    test_path = all_path[(model_test_rate-1)*splitLength:model_test_rate*splitLength-1]
    train_method = all_method[:(model_test_rate-1)*splitLength-1]+ all_method[model_test_rate*splitLength:]
    test_method = all_method[(model_test_rate-1)*splitLength:model_test_rate*splitLength-1]

# Remove words outside the vocabulary
updated_train_data = []    
updated_train_exception = []
updated_train_path = []
updated_train_method = []
updated_test_data = []
updated_test_exception = []
updated_test_path = []
updated_test_method = []

for j, item in enumerate(train_data):
    current_train_filter = [word for word in item if word in vocabulary]
    if (len(current_train_filter)>=min_sentence_len):
        updated_train_data.append(current_train_filter)
        updated_train_exception.append(train_exception[j])  
        updated_train_path.append(train_path[j])
        updated_train_method.append(train_method[j])

for j, item in enumerate(test_data):
    current_test_filter = [word for word in item if word in vocabulary]  
    if (len(current_test_filter)>=min_sentence_len):
        updated_test_data.append(current_test_filter)         
        updated_test_exception.append(test_exception[j])
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
        if sentence_len_count == max_sentence_len1-1:
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

print(input_filename1)
print("\nSentence length Average : %d\n"%(avg))

print("Under 10 : %d"%(under10))
print("Over 10, Under 200 : %d"%(over10under200))
print("Over 200, Under 400 : %d"%(over200under400))
print("Over 400 : %d\n"%(over400))

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
print("\nTest_zero: ",test_zero)
print("Train_zero: ",train_zero)
print("zero: ",test_zero+train_zero)
print("Test_one: ",test_one)
print("Train_one: ",train_one)
print("one: ",test_one+train_one)

limit_one = min(train_one,train_zero)
limit_zero = limit_one

final_train_data = []
final_train_exception = []
final_train_path = []
final_train_method = []

count_zero = 0
count_one = 0
for i, item in enumerate(updated_train_exception):
    if item == 0 and count_zero <= limit_zero:
        final_train_data.append(updated_train_data[i])
        final_train_exception.append(updated_train_exception[i])
        final_train_path.append(updated_train_path[i])
        final_train_method.append(updated_train_method[i])

        count_zero += 1
    elif item == 1 and count_one <= limit_one:
        final_train_data.append(updated_train_data[i])
        final_train_exception.append(updated_train_exception[i])
        final_train_path.append(updated_train_path[i])
        final_train_method.append(updated_train_method[i])

        count_one += 1
    if count_zero == limit_zero and count_one == limit_one:
        break

final_test_data = []
final_test_exception = []
final_test_path = []
final_test_method = []

limit_one = min(test_zero,test_one)
limit_zero = limit_one

count_zero = 0
count_one = 0
for i, item in enumerate(updated_test_exception):
    if item == 0 and count_zero <= limit_zero:
        final_test_data.append(updated_test_data[i])
        final_test_exception.append(updated_test_exception[i])
        final_test_path.append(updated_test_path[i])
        final_test_method.append(updated_test_method[i])

        count_zero += 1
    elif item == 1 and count_one <= limit_one:
        final_test_data.append(updated_test_data[i])
        final_test_exception.append(updated_test_exception[i])
        final_test_path.append(updated_test_path[i])
        final_test_method.append(updated_test_method[i])

        count_one += 1
    if count_zero == limit_zero and count_one == limit_one:
        break

# Create train and test data for deep learning + softmax
X_train = np.empty(shape=[len(final_train_data), max_sentence_len1, embed_size_word2vec], dtype='float32')
Y_train = np.empty(shape=[len(final_train_exception),1], dtype='int32')
# 1 - start of sentence, # 2 - end of sentence, # 0 - zero padding. Hence, word indices start with 3 
for j, curr_row in enumerate(final_train_data):
    sequence_cnt = 0         
    for item in curr_row:
        if item in vocabulary:
            X_train[j, sequence_cnt, :] = wordvec_model[item] 
            sequence_cnt = sequence_cnt + 1                
            if sequence_cnt == max_sentence_len1-1:
                break                
    for k in range(sequence_cnt, max_sentence_len1):
        X_train[j, k, :] = np.zeros((1,embed_size_word2vec))        
    Y_train[j,0] = final_train_exception[j]

X_test = np.empty(shape=[len(final_test_data), max_sentence_len1, embed_size_word2vec], dtype='float32')
Y_test = np.empty(shape=[len(final_test_exception),1], dtype='int32')
# 1 - start of sentence, # 2 - end of sentence, # 0 - zero padding. Hence, word indices start with 3 
for j, curr_row in enumerate(final_test_data):
    sequence_cnt = 0          
    for item in curr_row:
        if item in vocabulary:
            X_test[j, sequence_cnt, :] = wordvec_model[item] 
            sequence_cnt = sequence_cnt + 1                
            if sequence_cnt == max_sentence_len1-1:
                break                
    for k in range(sequence_cnt, max_sentence_len1):
        X_test[j, k, :] = np.zeros((1,embed_size_word2vec))        
    Y_test[j,0] = final_test_exception[j]
        
y_train = np_utils.to_categorical(Y_train, 2)
y_test = np_utils.to_categorical(Y_test, 2)

# Construct the deep learning model
'''
input1 = Input(shape=(max_sentence_len1, embed_size_word2vec), dtype='float32')
sequence = Masking(mask_value = 0.0)(input1)
forwards_1 = LSTM(LSTM_output_size, name='forwards_1')(sequence)
after_dp_forward_1 = Dropout(0.20, name='after_dp_forward_1')(forwards_1) 
backwards_1 = LSTM(LSTM_output_size, go_backwards=True, name='backwords_1')(sequence)
after_dp_backward_1 = Dropout(0.20, name='after_dp_backward_1')(backwards_1)         
merged_1 = keras.layers.concatenate([after_dp_forward_1, after_dp_backward_1], axis=-1)
after_dp_1= Dropout(0.5, name='after_dp_1')(merged_1)

output = Dense(2, activation='softmax', name='output')(after_dp_1)  
'''

model_json = "../JavaAutoException/Model/real_final/"+str(model_test_rate)+"/JAE_"+model_filename+"_"+str(model_count)+"_model.json"
model_h5 = "../JavaAutoException/Model/real_final/"+str(model_test_rate)+"/JAE_"+model_filename+"_"+str(model_count)+"_model.h5"
json_file = open(model_json, "r")
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights(model_h5)
print("Loaded model1 from disk")

rms = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08)    

# TRAIN MODEL
model.compile(loss='categorical_crossentropy', optimizer=rms, metrics=['accuracy'])

print("Count model1 parameter.")
model.count_params()
print("Get a short summary of each layer dimensions and parameters.")
model.summary()

roc_count = 1

predict = model.predict(X_test)

'''
preds = predict[:,1]
fpr, tpr, threshold = roc_curve(final_test_exception, preds)
roc_auc = auc(fpr, tpr)
'''
'''
nsamples,nx,ny = X_train.shape
d2_train_dataset = X_train.reshape((nsamples, nx*ny))
nsamples,nx,ny = X_test.shape
d2_test_dataset = X_test.reshape((nsamples, nx*ny))

ann_estimator = KerasRegressor(build_fn = model, epochs = 1, batch_size=batch_size, verbose=2)

boosted_ann = AdaBoostRegressor(base_estimator= ann_estimator)
#boosted_ann.fit(rescaledX, y_train.values.ravel())# scale your training data 
boosted_ann.fit(d2_train_dataset, Y_train, sample_weight=None)# scale your training data 
#boosted_ann.predict(rescaledX_Test)
preds = boosted_ann.predict(d2_test_dataset)
print(preds.shape)
print(preds)
sys.exit(1)
'''


bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), algorithm="SAMME", n_estimators=200)
#bdt = AdaBoostClassifier(model, algorithm="SAMME", n_estimators=1)

nsamples,nx,ny = X_train.shape
d2_train_dataset = X_train.reshape((nsamples, nx*ny))
nsamples,nx,ny = X_test.shape
d2_test_dataset = X_test.reshape((nsamples, nx*ny))

bdt.fit(d2_train_dataset, Y_train)

y_proba = bdt.predict_proba(d2_test_dataset)
predictY = np.argmax(y_proba, axis=1)

corrects = np.nonzero(predictY.reshape((-1,1)) == Y_test)
accuracy = len(corrects[0])/len(X_test)*100

preds = y_proba[:,1]
fpr, tpr, threshold = roc_curve(final_test_exception, preds)
roc_auc = auc(fpr, tpr)

idx = 0
test_one = 0
test_zero = 0
choose_one = 0
choose_zero = 0

for i, data in enumerate(final_test_exception):
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
cm = confusion_matrix(Y_test, predictY)
print(cm)
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
for p in range(cm.shape[0]):
    print('True label', p)
    for q in range(cm.shape[0]):
        print(cm[p,q], end=' ')
        if q%100 == 0:
            print(' ')
    print(' ')
''' 
train_result = hist.history
print("\nTrain_result", train_result)

true = np.nonzero(predictY.reshape((-1,1)) == y_test)
print(true)
accuracy = len(true[0])/len(predict)*100
train_zero = len(np.where(Y_train == 0))
train_one = len(np.where(Y_train == 1))
test_zero = len(np.where(Y_test == 0))
test_one = len(np.where(Y_test == 1))
choose_zero = len(np.where(predictY == 0))
choose_one = len(np.where(predictY == 1))
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

sys.exit(1)
'''
#TP, FP, TN, FN file save.
TP_file_name = "Code_snippet/Exception/"+str(model_test_rate)+"/TP_"+filename+"_"+str(count)+".txt"
FP_file_name = "Code_snippet/Exception/"+str(model_test_rate)+"/FP_"+filename+"_"+str(count)+ ".txt"
TN_file_name = "Code_snippet/Exception/"+str(model_test_rate)+"/TN_"+filename+"_"+str(count)+ ".txt"
FN_file_name = "Code_snippet/Exception/"+str(model_test_rate)+"/FN_"+filename+"_"+str(count)+ ".txt"

f_TP = open(TP_file_name, 'w')
f_FP = open(FP_file_name, 'w')
f_TN = open(TN_file_name, 'w')
f_FN = open(FN_file_name, 'w')

for i, data in enumerate(final_test_exception):
    if data == 1:
        if predictY[i] == 1:
            f_TP.write(final_test_path[i] + '\r\n')
            f_TP.write(final_test_method[i]+ '\r\n')
            f_TP.write(str(y_proba[i,0]) + '\t' + str(y_proba[i,1]) + '\r\n')
        else:
            f_FN.write(final_test_path[i]+ '\r\n')
            f_FN.write(final_test_method[i]+ '\r\n')
            f_FN.write(str(y_proba[i,0]) + '\t' + str(y_proba[i,1]) + '\r\n')
    else :
        if predictY[i] == 1:
            f_FP.write(final_test_path[i]+ '\r\n')
            f_FP.write(final_test_method[i]+ '\r\n')
            f_FP.write(str(y_proba[i,0]) + '\t' + str(y_proba[i,1]) + '\r\n')
        else:
            f_TN.write(final_test_path[i]+ '\r\n')
            f_TN.write(final_test_method[i]+ '\r\n')
            f_TN.write(str(y_proba[i,0]) + '\t' + str(y_proba[i,1]) + '\r\n')

f_TP.close()
f_FP.close()
f_TN.close()
f_FN.close()
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
roc_name = "ROC/"+str(model_test_rate) +"/JAE_"+filename+"_"+str(count)
plt.savefig(roc_name)
roc_count += 1
'''
'''
model_json = model.to_json()
model_name = "Model/Exception/"+str(model_test_rate) + "/JAE_"+filename+"_"+str(count)+"_model.json"
weight_name = "Model/Exception/"+str(model_test_rate) + "/JAE_"+filename+"_"+str(count)+"_model.h5"
with open(model_name,"w") as json_file :
    json_file.write(model_json)
model.save_weights(weight_name)
print("Saved model to disk\n\n\n")
'''
del bdt
