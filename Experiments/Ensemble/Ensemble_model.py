# Required dependencies
# 1. NLTK
# 2. Gensim for word2vec
# 3. Keras with tensorflow/theano backend

#=======================
import json
import codecs
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import keras
import keras.backend as K
from keras.layers import Reshape, Concatenate, Activation, LeakyReLU, Lambda, Flatten
from gensim.test.utils import common_texts, get_tmpfile #10-11
import gc
from sklearn import preprocessing
import pandas as pd
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords 
from nltk.tokenize import TweetTokenizer
from math import exp, log, e
import tensorflow as tf
from keras import metrics
from IPython.display import Image
#========================

import numpy as np
np.random.seed(1398)
import json, re, nltk, string
from nltk.corpus import wordnet
from gensim.models import Word2Vec
from keras.preprocessing import sequence
from keras.models import Model, model_from_json
from keras.layers import Dense, Dropout, Embedding, LSTM, Input, merge, Masking, TimeDistributed, Permute, Reshape, multiply
from keras.optimizers import RMSprop, Adam
from keras.utils import np_utils, plot_model
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report, confusion_matrix


#========================================================================================
## Motivation // only STAT -> MLP, Scikit learn
#========================================================================================
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import MultinomialNB, ComplementNB, GaussianNB

train_balancing=True
test_balancing=False
# 1 total act 偎熱, 2 user_comment 偎熱, 5 bin 偎熱, 7 recent activity
data_name = {mozilla:'mozilla', chrome:'chrome', firefox:'firefox', eclipse:'eclipse'}
#[0]:workday [1]:openday [2]:activityCnt [3]:cc, [4]:total act 偎熱, [5]:user_comm 偎熱
total_split = {'firefox': [[3,7],[1,2,3,10,30],[3,7,20,50],[1,2,3],[1,2,8,16],[1,2,4,8]], 'chrome':[[3,7],[1,2,3,10,30],[1,2,5,10],[1,2,3],[1,2,4,8],[1,2,4,8]], 'eclipse':[[3,7],[1,2,3,10,30],[1,2,3,5],[1,2,3],[1,2,4,8],[1,2,4,8]]} 

time_feature = False
test_mode = False

embedding_concat=False
one_hot_concat = True
system_log = False

mlp = False
scikit_learn = True
class_std = 2 

max_activity_len = 10
min_activity_len = 1 # workday 1-10
embedding_size=50

for i in range(6,11):
    totalLength = len(all_data)
    print('Total length: ', totalLength)
    splitLength = int(totalLength / (numCV + 1))
    print('class_std:',class_std)
    # Split cross validation set
    print ('CV',i)
    #========================================================================================
    # TRAIN DATA
    #========================================================================================
    if test_mode:
        train_data = all_data[:30]#i*splitLength]
        train_3to1 = sum([len(r) for r in train_data])       
        train_time = all_time[:train_3to1]#i*splitLength]
        train_totalact = all_totalact[:train_3to1]
        train_commcnt = all_commcnt[:train_3to1]
        train_workday = all_workday[:train_3to1]
        train_recentday = all_recentday[:train_3to1]
    else:
        train_data = all_data[:i*splitLength]
        train_3to1 = sum([len(r) for r in train_data])       
        train_time = all_time[:train_3to1]#i*splitLength]
        train_totalact = all_totalact[:train_3to1]
        train_commcnt = all_commcnt[:train_3to1]
        train_workday = all_workday[:train_3to1]
        train_recentday = all_recentday[:train_3to1]
     
    # ===================================================================== 
    updated_train_data = []    
    updated_train_history = []    
    updated_train_time = []
    
    j=0
    for bug1 in train_data:
        train_data_list = []
        train_history_list = []
        if len(bug1)>=min_activity_len:
            for act1 in bug1:
                current_train_filter = [word for word in act1 if word in vocabulary]
                train_data_list.append(current_train_filter)
                updated_train_time.append(train_time[j])
                j+=1
            updated_train_data.append(train_data_list)
        else:
            j+=len(bug1)
    
    del train_data, train_time, #train_bugid, train_workday
    gc.collect()
    
    # ===================================================================== 
    updated_train_time = [0 if x<=class_std else 1 for x in updated_train_time]
    curr_split = total_split[data_name[path]][0]
    updated_train_workday = [0 if x<=curr_split[0] else 1 if x<=curr_split[1] else 2 for x in train_workday]
    updated_train_recentday = []
    for x in train_recentday:
        tmp = 0 
        for j,y in enumerate(x):
            tmp+= pow(2,j)*y
        updated_train_recentday.append(tmp) 
    curr_split = total_split[data_name[path]][4]
    updated_train_totalact = [0 if x<=curr_split[0] else 1 if x<=curr_split[1] else 2 if x<=curr_split[2] else 3 if x<=curr_split[3] else 4 for x in train_totalact]
    curr_split = total_split[data_name[path]][5]
    updated_train_commcnt = [0 if x<=curr_split[0] else 1 if x<=curr_split[1] else 2 if x<=curr_split[2] else 3 if x<=curr_split[3] else 4 for x in train_commcnt]
    
    train_col_size = [len(x) for x in updated_train_data]
    train_n = len(updated_train_time)
    label_num = 2
    del updated_train_data, train_workday, train_recentday, train_totalact, train_commcnt
    gc.collect()
    
    # ===================================================================== 
    Y_train = np.asarray(updated_train_time, dtype=np.int32).reshape((-1,1))
    y_train = np_utils.to_categorical(Y_train, label_num)
    
    X_train_workday = np.asarray(updated_train_workday, dtype=np.float32)
    class_num = len(total_split[data_name[path]][0])+1
    X_train_workday = np_utils.to_categorical(X_train_workday, class_num)
    X_train_recentday = np.asarray(updated_train_recentday, dtype=np.float32)
    class_num = 8
    X_train_recentday = np_utils.to_categorical(X_train_recentday, class_num)
    X_train_totalact = np.asarray(updated_train_totalact, dtype=np.float32)
    class_num = len(total_split[data_name[path]][4])+1
    X_train_totalact = np_utils.to_categorical(X_train_totalact, class_num)
    X_train_commcnt = np.asarray(updated_train_commcnt, dtype=np.float32)
    class_num = len(total_split[data_name[path]][5])+1
    X_train_commcnt = np_utils.to_categorical(X_train_commcnt, class_num)

    X_train_concat = np.concatenate((X_train_workday, X_train_recentday),axis=1)
    X_train_concat = np.concatenate((X_train_concat, X_train_totalact),axis=1)
    X_train_concat = np.concatenate((X_train_concat, X_train_commcnt),axis=1)
    
    #========================================================================================
    # TEST DATA
    #========================================================================================
    if test_mode:
        test_data = all_data[30:50]#i*splitLength:(i+1)*splitLength] 
        test_3to1 = sum([len(r) for r in test_data])
        test_time = all_time[train_3to1:train_3to1+test_3to1]#i*splitLength:(i+1)*splitLength]
        #test_stream = all_stream[train_3to1 : train_3to1+test_3to1]
        test_totalact = all_totalact[train_3to1:train_3to1+test_3to1]
        test_commcnt = all_commcnt[train_3to1:train_3to1+test_3to1]
        test_workday = all_workday[train_3to1:train_3to1+test_3to1]
        test_recentday = all_recentday[train_3to1:train_3to1+test_3to1]
    else:
        test_data = all_data[i*splitLength:(i+1)*splitLength]
        test_3to1 = sum([len(r) for r in test_data])       
        test_time = all_time[train_3to1 : train_3to1+test_3to1]#i*splitLength]
        test_totalact = all_totalact[train_3to1:train_3to1+test_3to1]
        test_commcnt = all_commcnt[train_3to1:train_3to1+test_3to1]
        test_workday = all_workday[train_3to1:train_3to1+test_3to1]
        test_recentday = all_recentday[train_3to1:train_3to1+test_3to1]
        
    updated_test_data = []
    updated_test_time = []
    #updated_test_stream = []
    #updated_test_workday = []
    #updated_test_bugid = []
    
    j=0
    for bug1 in test_data:
        test_data_list = []
        test_history_list = []
        if len(bug1)>=min_activity_len:
            for act1 in bug1:
                current_test_filter = [word for word in act1 if word in vocabulary]
                test_data_list.append(current_test_filter)
                updated_test_time.append(test_time[j])
                #updated_test_stream.append(test_stream[j])
                #updated_test_workday.append(test_workday[j])
                #updated_test_bugid.append(test_bugid[j])
                j+=1
        else:
            j+=len(bug1)
        
    del test_data, test_time, # test_stream, test_bugid
    gc.collect()
    
    # ===================================================================== 
    updated_test_time = [0 if x<=class_std else 1 for x in updated_test_time]
    curr_split = total_split[data_name[path]][0]
    updated_test_workday = [0 if x<=curr_split[0] else 1 if x<=curr_split[1] else 2 for x in test_workday]
    updated_test_recentday = []
    for x in test_recentday:
        tmp = 0 
        for j,y in enumerate(x):
            tmp+= pow(2,j)*y
        updated_test_recentday.append(tmp) 
    curr_split = total_split[data_name[path]][4]
    updated_test_totalact = [0 if x<=curr_split[0] else 1 if x<=curr_split[1] else 2 if x<=curr_split[2] else 3 if x<=curr_split[3] else 4 for x in test_totalact]
    curr_split = total_split[data_name[path]][5]
    updated_test_commcnt = [0 if x<=curr_split[0] else 1 if x<=curr_split[1] else 2 if x<=curr_split[2] else 3 if x<=curr_split[3] else 4 for x in test_commcnt]
    
    test_col_size = [len(x) for x in updated_test_data]
    test_n = len(updated_test_time)
    del updated_test_data
    gc.collect()
    
    # ===================================================================== 
    y_test = np.asarray(updated_test_time, dtype=np.int32)
    y_test=y_test.reshape((-1,1))
    
    X_test_workday = np.asarray(updated_test_workday, dtype=np.float32)
    class_num = len(total_split[data_name[path]][0])+1
    X_test_workday = np_utils.to_categorical(X_test_workday, class_num)
    X_test_recentday = np.asarray(updated_test_recentday, dtype=np.float32)
    class_num = 8
    X_test_recentday = np_utils.to_categorical(X_test_recentday, class_num)
    X_test_totalact = np.asarray(updated_test_totalact, dtype=np.float32)
    class_num = len(total_split[data_name[path]][4])+1
    X_test_totalact = np_utils.to_categorical(X_test_totalact, class_num)
    X_test_commcnt = np.asarray(updated_test_commcnt, dtype=np.float32)
    class_num = len(total_split[data_name[path]][5])+1
    X_test_commcnt = np_utils.to_categorical(X_test_commcnt, class_num)

    X_test_concat = np.concatenate((X_test_workday, X_test_recentday),axis=1)
    X_test_concat = np.concatenate((X_test_concat, X_test_totalact),axis=1)
    X_test_concat = np.concatenate((X_test_concat, X_test_commcnt),axis=1)
    
    # ===================================================================== 
    label_num=max(updated_train_time)+1
    if mlp:
        embedding_input=Input(shape=(len(X_train_concat[0]),), dtype='float32')

        middle = Dense(10)(embedding_input)
        middle = LeakyReLU(alpha=0.3)(middle)
        middle = Dense(15)(middle)
        middle = LeakyReLU(alpha=0.2)(middle)
        #middle = Dropout(0.5)(middle)
        middle = Dense(8)(middle)
        middle = LeakyReLU(alpha=0.5)(middle)

        output = Dense(label_num, activation='softmax')(middle)

        model = Model(input=[embedding_input], output=output)
        adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy']) 
        model.summary()

        # =====================================================================
        # TRAIN MODEL
        es = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto')
        X_train_concat, X_valid_concat, y_train, y_valid = train_test_split(X_train_concat, y_train, test_size=0.15, shuffle= True)
        hist = model.fit(X_train_concat, y_train, batch_size=batch_size, 
                         validation_data = (X_valid_concat, y_valid), epochs=20, callbacks=[es])
        train_result = hist.history
        print('\nTrain_result\n')
        print(train_result)
        del X_train_concat, X_valid_concat, y_train, y_valid
        gc.collect()
        
        #========================================================================================
        # PREDICT & ACCURACY
        #========================================================================================
        predict = model.predict([X_test_concat]) 
        predictY = np.argmax(predict, axis=1)
        corrects = np.nonzero(predictY.reshape((-1,1)) == y_test) #.reshape((-1,))
        print("Total test accuracy : %.2f" % (len(corrects[0])/len(X_test_concat)*100))
        print('\nConfusion Matrix')
        cm = confusion_matrix(y_test, predictY)
        print(cm)
        print('\nf1_score: %.2f' % (f1_score(y_test, predictY, average="macro")*100))
        print('precision_score: %.2f'% (precision_score(y_test, predictY, average="macro")*100)) # class 0 accuracy
        print('recall_score: %.2f' % (recall_score(y_test, predictY, average="macro")*100)) # class 1 accuracy
        total = sum(cm[0])+sum(cm[1])
        print('Predict 0: %.2f' % ((cm[0,0]+cm[1,0])/total*100))
        print('Predict 1: %.2f' % ((cm[0,1]+cm[1,1])/total*100))
        print()
        del model
        gc.collect()
        
    elif scikit_learn:
        # 1. Logistic Regression
        print('================ 1. Logistic Regression ================')
        log_reg = LogisticRegression(penalty="l2")
        log_reg.fit(X_train_concat,Y_train) 
        y_proba = log_reg.predict_proba(X_test_concat)
        y_proba = np.argmax(y_proba, axis=1)
        corrects = np.nonzero(y_proba.reshape((-1,1)) == y_test) #.reshape((-1,))
        print("Total test accuracy : %.2f" % (len(corrects[0])/len(X_test_concat)*100))
        print('\nConfusion Matrix')
        cm = confusion_matrix(y_test, y_proba)
        print(cm)
        print('\nf1_score: %.2f' % (f1_score(y_test, y_proba, average="macro")*100))
        print('precision_score: %.2f'% (precision_score(y_test, y_proba, average="macro")*100)) # class 0 accuracy
        print('recall_score: %.2f' % (recall_score(y_test, y_proba, average="macro")*100)) # class 1 accuracy
        total = sum(cm[0])+sum(cm[1])
        print('Predict 0: %.2f' % ((cm[0,0]+cm[1,0])/total*100))
        print('Predict 1: %.2f' % ((cm[0,1]+cm[1,1])/total*100))
        print()

        '''
        # 2. SVM
        print('================ 2. SVM ================')
        clf = svm.SVC(gamma='scale', decision_function_shape='ovo', probability=True)
        clf.fit(X_train_concat,y_train) 
        y_proba = clf.predict_proba(X_test_concat)
        y_proba = np.argmax(y_proba, axis=1)
        corrects = np.nonzero(y_proba.reshape((-1,1)) == y_test) #.reshape((-1,))
        print("Total test accuracy : %.2f" % (len(corrects[0])/len(X_test_concat)*100))
        print('\nConfusion Matrix')
        cm = confusion_matrix(y_test, y_proba)
        print(cm)
        print('\nf1_score: %.2f' % (f1_score(y_test, y_proba, average="macro")*100))
        print('precision_score: %.2f'% (precision_score(y_test, y_proba, average="macro")*100)) # class 0 accuracy
        print('recall_score: %.2f' % (recall_score(y_test, y_proba, average="macro")*100)) # class 1 accuracy
        total = sum(cm[0])+sum(cm[1])
        print('Predict 0: %.2f' % ((cm[0,0]+cm[1,0])/total*100))
        print('Predict 1: %.2f' % ((cm[0,1]+cm[1,1])/total*100))
        print()
        del clf
        gc.collect()
        '''
        # 3. Naive Bayes
        print('================ 3. Naive Bayes ================')
        clf = GaussianNB()
        clf.fit(X_train_concat,Y_train) 
        y_proba = clf.predict_proba(X_test_concat)
        y_proba = np.argmax(y_proba, axis=1)
        corrects = np.nonzero(y_proba.reshape((-1,1)) == y_test) #.reshape((-1,))
        print("Total test accuracy : %.2f" % (len(corrects[0])/len(X_test_concat)*100))
        print('\nConfusion Matrix')
        cm = confusion_matrix(y_test, y_proba)
        print(cm)
        print('\nf1_score: %.2f' % (f1_score(y_test, y_proba, average="macro")*100))
        print('precision_score: %.2f'% (precision_score(y_test, y_proba, average="macro")*100)) # class 0 accuracy
        print('recall_score: %.2f' % (recall_score(y_test, y_proba, average="macro")*100)) # class 1 accuracy
        total = sum(cm[0])+sum(cm[1])
        print('Predict 0: %.2f' % ((cm[0,0]+cm[1,0])/total*100))
        print('Predict 1: %.2f' % ((cm[0,1]+cm[1,1])/total*100))
        print()
        del clf
        gc.collect()


        # 4. Random Forest
        print('================ 4. Random Forest ================')
        clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
        clf.fit(X_train_concat,Y_train) 
        y_proba = clf.predict_proba(X_test_concat)
        y_proba = np.argmax(y_proba, axis=1)
        corrects = np.nonzero(y_proba.reshape((-1,1)) == y_test) #.reshape((-1,))
        print("Total test accuracy : %.2f" % (len(corrects[0])/len(X_test_concat)*100))
        print('\nConfusion Matrix')
        cm = confusion_matrix(y_test, y_proba)
        print(cm)
        print('\nf1_score: %.2f' % (f1_score(y_test, y_proba, average="macro")*100))
        print('precision_score: %.2f'% (precision_score(y_test, y_proba, average="macro")*100)) # class 0 accuracy
        print('recall_score: %.2f' % (recall_score(y_test, y_proba, average="macro")*100)) # class 1 accuracy
        total = sum(cm[0])+sum(cm[1])
        print('Predict 0: %.2f' % ((cm[0,0]+cm[1,0])/total*100))
        print('Predict 1: %.2f' % ((cm[0,1]+cm[1,1])/total*100))
        print()
        del clf
        gc.collect()

        # 5. AdaBoost
        print('================ 5. AdaBoost ================')
        clf = AdaBoostClassifier(n_estimators=100)
        clf.fit(X_train_concat,Y_train) 
        y_proba = clf.predict_proba(X_test_concat)
        y_proba = np.argmax(y_proba, axis=1)
        corrects = np.nonzero(y_proba.reshape((-1,1)) == y_test) #.reshape((-1,))
        print("Total test accuracy : %.2f" % (len(corrects[0])/len(X_test_concat)*100))
        print('\nConfusion Matrix')
        cm = confusion_matrix(y_test, y_proba)
        print(cm)
        print('\nf1_score: %.2f' % (f1_score(y_test, y_proba, average="macro")*100))
        print('precision_score: %.2f'% (precision_score(y_test, y_proba, average="macro")*100)) # class 0 accuracy
        print('recall_score: %.2f' % (recall_score(y_test, y_proba, average="macro")*100)) # class 1 accuracy
        total = sum(cm[0])+sum(cm[1])
        print('Predict 0: %.2f' % ((cm[0,0]+cm[1,0])/total*100))
        print('Predict 1: %.2f' % ((cm[0,1]+cm[1,1])/total*100))
        print()
        del clf
        gc.collect()
    
    
