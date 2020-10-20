# Required dependencies
# 1. NLTK
# 2. Gensim for word2vec
# 3. Keras with tensorflow/theano backend
import preprocessing_data as pr
import model as md
import evaluation as ev

import sys
import gc
from copy import deepcopy
from keras.callbacks import ModelCheckpoint, EarlyStopping

input_file_num = int(sys.argv[2])
#========================================================================================
# Initializing Hyper parameter
#========================================================================================
#1. Word2vec parameters
min_word_frequency_word2vec = int(sys.argv[input_file_num+4])
embed_size_word2vec = 200
context_window_word2vec = 5

#2. Classifier hyperparameters
st_min_sentence_len = int(sys.argv[input_file_num+6])
st_max_sentence_len = int(sys.argv[input_file_num+7])
batch_size = 64
LSTM_output_size = 64
epoch_len = 15
test_rate = int(sys.argv[input_file_num+3])
label = sys.argv[input_file_num+8]
if label == "Logging":
    label_name = "isLogged"
elif label == "Exception":
    label_name = "isException"

#3. Input file name
input_file = sys.argv[1]
input_filename = []
for i in range(input_file_num):
    input_filename.append(sys.argv[i+3])
    input_filename[i] = input_file + '-' + input_filename[i]
if input_file_num == 1:
    filename = input_filename[0] + "_min"+str(st_min_sentence_len)+"_max"+str(st_max_sentence_len)+"_frequency"+str(min_word_frequency_word2vec)
    wordvec_model_name = input_filename[0] + "_frequency"+str(min_word_frequency_word2vec)
else:
    filename = input_file + "-CAST_m_"
    for i in range(input_file_num):
        filename += input_filename[i][len(input_file)+1]
    filename += "_min"+str(st_min_sentence_len)+'_max'+str(st_max_sentence_len)+'_frequency'+str(min_word_frequency_word2vec)
    wordvec_model_name = input_file + "-CAST_m_"
    for i in range(input_file_num):
        wordvec_model_name += input_filename[i][len(input_file)+1]
    wordvec_model_name += '_frequency'+str(min_word_frequency_word2vec)

#========================================================================================
# Preprocess the java auto logging, extract the vocabulary and learn the word2vec representation
#========================================================================================

all_data = [[] for _ in range(input_file_num)]
all_label = [[] for _ in range(input_file_num)]
all_path = [[] for _ in range(input_file_num)]
all_method = [[] for _ in range(input_file_num)]
for r in range(input_file_num):
    for i in range(st_min_sentence_len, st_max_sentence_len+1):
        json_file_path = "../%s/Data/sentence_len/data2/%s/%s_sentence_len%s_frequency%s.json"%(label, input_file, input_filename[r], str(i), str(min_word_frequency_word2vec))
        temp_data, temp_label, temp_path, temp_method = pr.open_data(json_file_path, label_name)

        all_data[r] += temp_data
        all_label[r] += temp_label
        all_path[r] += temp_path
        all_method[r] += temp_method

    zero, one = ev.return_the_number_of_label_data(all_label[r])
    print("%sth data"%(r+1))
    print("zero : ", zero)
    print("one : ", one)
    print()

# Learn the word2vec model and extract vocabulary
model_name = "../%s/Wordvec_Model/"%(label)+wordvec_model_name+".model"
wordvec_model, vocabulary = pr.load_word2vec_model(model_name)

#========================================================================================
# Split cross validation sets and perform deep learning + softmax based classification
#========================================================================================

seed = int(sys.argv[input_file_num+5])
nth_fold = int(sys.argv[input_file_num+9])

train_data = [0 for _ in range(input_file_num)]
test_data = [0 for _ in range(input_file_num)]
train_label= [0 for _ in range(input_file_num)]
test_label = [0 for _ in range(input_file_num)]
train_path = [0 for _ in range(input_file_num)]
test_path = [0 for _ in range(input_file_num)]
train_method = [0 for _ in range(input_file_num)]
test_method = [0 for _ in range(input_file_num)]

max_sentence_len = [0 for _ in range(input_file_num)]

X_train = [0 for _ in range(input_file_num)]
X_test = [0 for _ in range(input_file_num)]

for r in range(input_file_num):
    pr.data_shuffle(all_data[r], seed)
    pr.data_shuffle(all_label[r], seed)
    pr.data_shuffle(all_path[r], seed)
    pr.data_shuffle(all_method[r], seed)
   
    # Split cross validation set
    train_data[r], test_data[r] = pr.split_train_test_data(test_rate, all_data[r], nth_fold)
    train_label[r], test_label[r] = pr.split_train_test_data(test_rate, all_label[r], nth_fold)
    train_path[r], test_path[r] = pr.split_train_test_data(test_rate, all_path[r], nth_fold)
    train_method[r], test_method[r] = pr.split_train_test_data(test_rate, all_method[r], nth_fold)

    zero, one = ev.return_the_number_of_label_data(train_label[r])
    print("Train")
    print("zero : ", zero)
    print("one : ", one)

    zero, one = ev.return_the_number_of_label_data(test_label[r])
    print("Test")
    print("zero : ", zero)
    print("one : ", one)
    print()

    pr.print_the_number_of_data_split_document_size(train_data[r]+test_data[r])

    max_sentence_len[r] = pr.check_max_sentence_len(train_data[r] + test_data[r], train_path[r]+test_path[r], train_method[r]+test_method[r])
    print(max_sentence_len[r])

    # Make model input.
    X_train[r], Y_train = pr.create_deep_learning_input_data(train_data[r], train_label[r], max_sentence_len[r], embed_size_word2vec, vocabulary, wordvec_model)
    X_test[r], Y_test = pr.create_deep_learning_input_data(test_data[r], test_label[r], max_sentence_len[r], embed_size_word2vec, vocabulary, wordvec_model)
quit()
'''
# Save input data to json format.
input_train_data_path = "../input_data/train/"+str(test_rate)+"/"+filename+".json"
input_test_data_path = "../input_data/test/"+str(test_rate)+"/"+filename+".json"
pr.make_input_merge_data_json(input_train_data_path, label_name, merge_train_data, train_logging, train_path, train_method)
pr.make_input_merge_data_json(input_test_data_path, label_name, merge_test_data, test_logging, test_path, test_method)
'''

# Construct the deep learning model
if input_file_num == 1:
    model = md.bidirectional_RNN(max_sentence_len[0], embed_size_word2vec, LSTM_output_size)
elif input_file_num == 2:
    model = md.merge_2_bidirectional_RNN(max_sentence_len[0], max_sentence_len[1], embed_size_word2vec, LSTM_output_size)
elif input_file_num == 3:
    model = md.merge_3_bidirectional_RNN(max_sentence_len[0], max_sentence_len[1], max_sentence_len[2],  embed_size_word2vec, LSTM_output_size)
elif input_file_num == 4:
    model = md.merge_4_bidirectional_RNN(max_sentence_len[0], max_sentence_len[1], max_sentence_len[2], max_sentence_len[3], embed_size_word2vec, LSTM_output_size)
print("Count model parameter.")
model.count_params()
print("Get a short summary of each layer dimensions and parameters.")
model.summary()
'''
model_path = "Model/end2/"+str(test_rate) + "/best/JAL_"+filename+"_"+str(count)+"_model.hdf5"
callbacks = [EarlyStopping(monitor='val_acc',mode='max',verbose=1, patience=3), ModelCheckpoint(model_path, monitor='val_acc',verbose=1,mode='max', save_best_only=True, save_weights_only=False)]

if input_file_num == 1:
    hist = model.fit(X_train, Y_train, batch_size = batch_size, epochs = 20, verbose = 1, callbacks=callbacks)
    predict = model.predict(X_test)
elif input_file_num == 2:
    hist = model.fit([X_train, X_train2], Y_train, batch_size = batch_size, epochs=20, verbose=1, callbacks=callbacks)
    predict = model.predict([X_test, X_test2])
elif input_file_num == 3:
    hist = model.fit([X_train, X_train2, X_train3], Y_train, batch_size = batch_size, epochs=20, verbose=1, callbacks=callbacks)
    predict = model.predict([X_test, X_test2, X_test3])
elif input_file_num == 4:
    hist = model.fit([X_train, X_train2, X_train3, X_train4], Y_train, batch_size = batch_size, epochs=20, verbose=1, callbacks=callbacks)
    predict = model.predict([X_test, X_test2, X_test3, X_test4])

predict_1d = predict.argmax(axis = 1)
Y_train_1d = Y_train.argmax(axis = 1)
Y_test_1d = Y_test.argmax(axis = 1)
preds = predict[:,1]

accuracy = ev.accuracy(predict_1d, Y_test_1d)

train_zero, train_one = ev.return_the_number_of_label_data(Y_train_1d)
test_zero, test_one = ev.return_the_number_of_label_data(Y_test_1d)
choose_zero, choose_one = ev.return_the_number_of_predict_data(predict_1d)

# Print evaluate results.
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

f1_score = ev.f1_score(Y_test_1d, predict_1d)
print("\nF1score : ", f1_score)

roc_auc = ev.AUC(Y_test_1d, preds)
print("\nAUC : ", roc_auc)

# Print confusion matrix.
ev.print_confusion_matrix(Y_test_1d, predict_1d)

train_result = hist.history
print("\nTrain_result", train_result)

# Save predict rate with path and method of data.
TP_file_name = "Code_snippet/end2/"+str(test_rate)+"/TP_"+filename+"_"+str(count)+".txt"
FP_file_name = "Code_snippet/end2/"+str(test_rate)+"/FP_"+filename+"_"+str(count)+ ".txt"
TN_file_name = "Code_snippet/end2/"+str(test_rate)+"/TN_"+filename+"_"+str(count)+ ".txt"
FN_file_name = "Code_snippet/end2/"+str(test_rate)+"/FN_"+filename+"_"+str(count)+ ".txt"
ev.save_predict_rate(TP_file_name, FP_file_name, TN_file_name, FN_file_name, predict_1d, predict, Y_test_1d, test_path, test_method)
# Save ROC graph.
roc_name = "ROC/end2/"+str(test_rate) +"/JAL_"+filename+"_"+str(count)
ev.save_ROC(1, Y_test_1d, preds, roc_name)

# Save model.
model_name = "Model/end2/"+str(test_rate) + "/JAL_"+filename+"_"+str(count)+"_model.json"
weight_name = "Model/end2/"+str(test_rate) + "/JAL_"+filename+"_"+str(count)+"_model.h5"
md.save_model(model, model_name, weight_name)

del model
gc.collect()
'''

accuracy_list = []
roc_count = 1
for ep in range(epoch_len) :
    print(str(ep+1) + "\n")
    if input_file_num == 1:
        hist = model.fit(X_train[0], Y_train, batch_size = batch_size, epochs = 1, verbose = 2)
        predict = model.predict(X_test[0])
    elif input_file_num == 2:
        hist = model.fit([X_train[0], X_train[1]], Y_train, batch_size = batch_size, epochs=1, verbose=2)
        predict = model.predict([X_test[0], X_test[1]])
    elif input_file_num == 3:
        hist = model.fit([X_train[0], X_train[1], X_train[2]], Y_train, batch_size = batch_size, epochs=1, verbose=2)
        predict = model.predict([X_test[0], X_test[1], X_test[2]])
    elif input_file_num == 4:
        hist = model.fit([X_train[0], X_train[1], X_train[2], X_train[3]], Y_train, batch_size = batch_size, epochs=1, verbose=2)
        predict = model.predict([X_test[0], X_test[1], X_test[2], X_test[3]])

    predict_1d = predict.argmax(axis = 1)
    Y_train_1d = Y_train.argmax(axis = 1)
    Y_test_1d = Y_test.argmax(axis = 1)
    preds = predict[:,1]

    accuracy = ev.accuracy(predict_1d, Y_test_1d)
    accuracy_list.append([accuracy, ep+1])

    train_zero, train_one = ev.return_the_number_of_label_data(Y_train_1d)
    test_zero, test_one = ev.return_the_number_of_label_data(Y_test_1d)
    choose_zero, choose_one = ev.return_the_number_of_predict_data(predict_1d)

    # Print evaluate results.
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

    f1_score = ev.f1_score(Y_test_1d, predict_1d)
    print("\nF1score : ", f1_score)

    roc_auc = ev.AUC(Y_test_1d, preds)
    print("\nAUC : ", roc_auc)

    # Print confusion matrix.
    ev.print_confusion_matrix(Y_test_1d, predict_1d)

    train_result = hist.history
    print("\nTrain_result", train_result)

    '''
    # Save predict rate with path and method of data.
    TP_file_name = "../Code_snippet/"+str(test_rate)+"/TP_"+filename+"_ep"+str(ep+1)+".txt"
    FP_file_name = "../Code_snippet/"+str(test_rate)+"/FP_"+filename+"_ep"+str(ep+1)+".txt"
    TN_file_name = "../Code_snippet/"+str(test_rate)+"/TN_"+filename+"_ep"+str(ep+1)+".txt"
    FN_file_name = "../Code_snippet/"+str(test_rate)+"/FN_"+filename+"_ep"+str(ep+1)+".txt"
    ev.save_predict_rate(TP_file_name, FP_file_name, TN_file_name, FN_file_name, predict_1d, predict, Y_test_1d, test_path, test_method)
    '''
    # Save ROC graph.
    roc_name = "../%s/ROC/%s/%s_ep%s"%(label, str(test_rate), filename, str(ep+1))
    ev.save_ROC(roc_count, Y_test_1d, preds, roc_name)
    roc_count += 1

    # Save model.
    model_name = "../%s/Model/%s/%s_ep%s.json"%(label, str(test_rate), filename, str(ep+1))
    weight_name = "../%s/Model/%s/%s_ep%s.h5"%(label, str(test_rate), filename, str(ep+1))
    md.save_model(model, model_name, weight_name)

print(accuracy_list)
print("max accuracy : ", max(accuracy_list))
del model
gc.collect()
