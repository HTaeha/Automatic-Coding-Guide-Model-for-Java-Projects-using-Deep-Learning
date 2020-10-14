# Required dependencies
# 1. NLTK
# 2. Gensim for word2vec
# 3. Keras with tensorflow/theano backend
import preprocessing_data as pr
import model as md
import evaluation as ev

import sys
import numpy as np
from copy import deepcopy
from keras.models import model_from_json

input_file_num = int(sys.argv[4])
pred_by_sentence_len = False
#========================================================================================
# Initializing Hyper parameter
#========================================================================================
#1. Word2vec parameters
min_word_frequency_word2vec = int(sys.argv[input_file_num+6])
embed_size_word2vec = 200

#2. Classifier hyperparameters
numCV = 1
min_sentence_len = int(sys.argv[input_file_num+7])
max_sentence_len = int(sys.argv[input_file_num+8])
model_test_rate = int(sys.argv[input_file_num+5])
count = 1
model_count = 1
label_name = "isLogged"

#3. Input file name
target_file = sys.argv[1]
source_file = sys.argv[2]
input_filename = []
for i in range(input_file_num):
    input_filename.append(sys.argv[i+5])
    input_filename[i] = source_file + '-' + input_filename[i]
target_type = sys.argv[3]
target_filename = target_file + '-' + target_type
wordvec_model_name = target_filename + '_frequency'+str(min_word_frequency_word2vec)
model_filename = target_filename + "_sentence_balanced_min"+str(min_sentence_len)+"_max"+str(max_sentence_len)+"_frequency"+str(min_word_frequency_word2vec)

#standard_input_type = sys.argv[4]
standard_input_filename = source_file + '-code'
standard_wordvec_model_name = standard_input_filename + '_frequency'+str(min_word_frequency_word2vec)
#========================================================================================
# Preprocess the java auto logging, extract the vocabulary and learn the word2vec representation
#========================================================================================

for i in range(min_sentence_len, max_sentence_len+1):
    java_auto_logging_json = "input_data/end2/sentence_len/data1/"+standard_input_filename+"_sentence_len"+str(i)+"_frequency"+str(min_word_frequency_word2vec)+".json"
    temp_data, temp_logging, temp_path, temp_method = pr.open_data(java_auto_logging_json, label_name)
    if i == min_sentence_len:
        standard_data = deepcopy(temp_data)
        standard_logging = deepcopy(temp_logging)
        standard_path = deepcopy(temp_path)
        standard_method = deepcopy(temp_method)
    else:
        standard_data += temp_data
        standard_logging += temp_logging
        standard_path += temp_path
        standard_method += temp_method


zero, one = ev.return_the_number_of_label_data(standard_logging)
print("Standard data")
print("zero : ", zero)
print("one : ", one)
print()

if input_file_num >= 1:
    for i in range(min_sentence_len, max_sentence_len+1):
        java_auto_logging_json = "input_data/end2/sentence_len/data1/"+input_filename[0]+"_sentence_len"+str(i)+"_frequency"+str(min_word_frequency_word2vec)+".json"
        temp_data, temp_logging, temp_path, temp_method = pr.open_data(java_auto_logging_json, label_name)
        if i == min_sentence_len:
            all_data = deepcopy(temp_data)
            all_logging = deepcopy(temp_logging)
            all_path = deepcopy(temp_path)
            all_method = deepcopy(temp_method)
        else:
            all_data += temp_data
            all_logging += temp_logging
            all_path += temp_path
            all_method += temp_method

    zero, one = ev.return_the_number_of_label_data(all_logging)
    print("First data")
    print("zero : ", zero)
    print("one : ", one)
    print()

if input_file_num >= 2:
    for i in range(min_sentence_len, max_sentence_len+1):
        java_auto_logging_json = "input_data/end2/sentence_len/data1/"+input_filename[1]+"_sentence_len"+str(i)+"_frequency"+str(min_word_frequency_word2vec)+".json"
        temp_data, temp_logging, temp_path, temp_method = pr.open_data(java_auto_logging_json, label_name)
        if i == min_sentence_len:
            all_data2 = deepcopy(temp_data)
            all_logging2 = deepcopy(temp_logging)
            all_path2 = deepcopy(temp_path)
            all_method2 = deepcopy(temp_method)
        else:
            all_data2 += temp_data
            all_logging2 += temp_logging
            all_path2 += temp_path
            all_method2 += temp_method

    zero, one = ev.return_the_number_of_label_data(all_logging2)
    print("Second data")
    print("zero : ", zero)
    print("one : ", one)
    print()

if input_file_num >= 3:
    for i in range(min_sentence_len, max_sentence_len+1):
        java_auto_logging_json = "input_data/end2/sentence_len/data1/"+input_filename[2]+"_sentence_len"+str(i)+"_frequency"+str(min_word_frequency_word2vec)+".json"
        temp_data, temp_logging, temp_path, temp_method = pr.open_data(java_auto_logging_json, label_name)
        if i == min_sentence_len:
            all_data3 = deepcopy(temp_data)
            all_logging3 = deepcopy(temp_logging)
            all_path3 = deepcopy(temp_path)
            all_method3 = deepcopy(temp_method)
        else:
            all_data3 += temp_data
            all_logging3 += temp_logging
            all_path3 += temp_path
            all_method3 += temp_method

    zero, one = ev.return_the_number_of_label_data(all_logging3)
    print("Third data")
    print("zero : ", zero)
    print("one : ", one)
    print()

if input_file_num >= 4:
    for i in range(min_sentence_len, max_sentence_len+1):
        java_auto_logging_json = "input_data/end2/sentence_len/data1/"+input_filename[3]+"_sentence_len"+str(i)+"_frequency"+str(min_word_frequency_word2vec)+".json"
        temp_data, temp_logging, temp_path, temp_method = pr.open_data(java_auto_logging_json, label_name)
        if i == min_sentence_len:
            all_data4 = deepcopy(temp_data)
            all_logging4 = deepcopy(temp_logging)
            all_path4 = deepcopy(temp_path)
            all_method4 = deepcopy(temp_method)
        else:
            all_data4 += temp_data
            all_logging4 += temp_logging
            all_path4 += temp_path
            all_method4 += temp_method

    zero, one = ev.return_the_number_of_label_data(all_logging4)
    print("4th data")
    print("zero : ", zero)
    print("one : ", one)
    print()

# Load the word2vec model and vocabulary
model_name = "Wordvec_Model/end2/"+wordvec_model_name+"_"+str(model_count)+".model"
standard_model_name = "Wordvec_Model/end2/"+standard_wordvec_model_name+"_"+str(model_count)+".model"

wordvec_model, vocabulary = pr.load_word2vec_model(model_name)
standard_wordvec_model, standard_vocabulary = pr.load_word2vec_model(standard_model_name)

# Load model
model_json = "Model/end2/"+str(model_test_rate)+"/best/JAL_"+model_filename+"_"+str(model_count)+"_model.json"
model_h5 = "Model/end2/"+str(model_test_rate)+"/best/JAL_"+model_filename+"_"+str(model_count)+"_model.h5"
model = md.load_model(model_json, model_h5)
'''
json_file = open(model_json, "r")
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights(model_h5)
print("Loaded model from disk")
'''
#========================================================================================
# Split cross validation sets and perform deep learning + softmax based classification
#========================================================================================
pr.data_shuffle(standard_data, 1398)

merge_train_data = []
merge_test_data = []
if input_file_num >= 1:
    pr.data_shuffle(all_data, 1398)
    pr.data_shuffle(all_logging, 1398)
    pr.data_shuffle(all_path, 1398)
    pr.data_shuffle(all_method, 1398)
   
    # Set data document size.
#all_data, all_logging, all_path, all_method = pr.set_document_size2(standard_data, all_data, all_logging, all_path, all_method, min_sentence_len, max_sentence_len)

    # Split cross validation set
    standard_train_data, standard_test_data = pr.split_train_test_data(model_test_rate, standard_data, 10)
    standard_train_logging, standard_test_logging = pr.split_train_test_data(model_test_rate, standard_logging, 10)

    '''
    train_data, test_data = pr.split_train_test_data(model_test_rate, all_data, 10)
    train_logging, test_logging = pr.split_train_test_data(model_test_rate, all_logging, 10)
    train_path, test_path = pr.split_train_test_data(model_test_rate, all_path, 10)
    train_method, test_method = pr.split_train_test_data(model_test_rate, all_method, 10)
    '''

    '''
    zero, one = ev.return_the_number_of_label_data(train_logging)
    print(input_filename[0])
    print("After set document size of train data, the number of zero and one label data : ",zero, one)
    train_limit_num = min(zero, one)
    zero, one = ev.return_the_number_of_label_data(test_logging)
    print("After set document size of test data, the number of zero and one label data : ",zero, one)
    test_limit_num = min(zero, one)
    '''

    zero, one = ev.return_the_number_of_label_data(all_logging)
    print("The number of zero and one label data : ",zero, one)

    pr.print_the_number_of_data_split_document_size(all_data)
    # Balance out the number of data.
#   train_data, train_logging, train_path, train_method = pr.balance_out_data(train_data, train_logging, train_path, train_method, train_limit_num)
#   test_data, test_logging, test_path, test_method = pr.balance_out_data(test_data, test_logging, test_path, test_method, test_limit_num)

    # Remove words outside the vocabulary
    standard_test_data = pr.remove_words_outside_the_vocabulary(standard_test_data, standard_vocabulary)

    test_data = pr.remove_words_outside_the_vocabulary(all_data, vocabulary)
    test_logging = deepcopy(all_logging)
    test_path = deepcopy(all_path)
    test_method = deepcopy(all_method)

    for layer in model.layers:
        if layer.name == "input_1":
            max_sentence_len = layer.get_output_at(0).get_shape().as_list()[1]
            break

#max_sentence_len = model.layers[0].get_output_at(0).get_shape().as_list()[1]
    print(max_sentence_len)

    # Make model input.
    st_X_test, st_Y_test = pr.create_deep_learning_input_data(standard_test_data, standard_test_logging, max_sentence_len, embed_size_word2vec, standard_vocabulary, standard_wordvec_model)

    X_test, Y_test = pr.create_deep_learning_input_data(test_data, test_logging, max_sentence_len, embed_size_word2vec, vocabulary, wordvec_model)

if input_file_num >= 2:
    pr.data_shuffle(all_data2, 1398)
    pr.data_shuffle(all_logging2, 1398)
    pr.data_shuffle(all_path2, 1398)
    pr.data_shuffle(all_method2, 1398)

    # Set data document size.
#all_data2, all_logging2, all_path2, all_method2 = pr.set_document_size2(standard_data,all_data2,all_logging2, all_path2, all_method2, min_sentence_len, max_sentence_len)

    # Split cross validation set
    '''
    train_data2, test_data2 = pr.split_train_test_data(model_test_rate, all_data2, 10)
    train_logging2, test_logging2 = pr.split_train_test_data(model_test_rate, all_logging2, 10)
    train_path2, test_path2 = pr.split_train_test_data(model_test_rate, all_path2, 10)
    train_method2, test_method2 = pr.split_train_test_data(model_test_rate, all_method2, 10)
    '''
    '''
    zero, one = ev.return_the_number_of_label_data(train_logging2)
    print(input_filename[1])
    print("After set document size of train data, the number of zero and one label data : ",zero, one)
    train_limit_num2 = min(zero, one)
    zero, one = ev.return_the_number_of_label_data(test_logging2)
    print("After set document size of test data, the number of zero and one label data : ",zero, one)
    test_limit_num2 = min(zero, one)
    '''

    zero, one = ev.return_the_number_of_label_data(all_logging2)
    print("The number of zero and one label data : ",zero, one)
    # Balance out the number of data.
#   train_data2, train_logging2, train_path2, train_method2 = pr.balance_out_data(train_data2, train_logging2, train_path2, train_method2, train_limit_num2)
#   test_data2, test_logging2, test_path2, test_method2 = pr.balance_out_data(test_data2, test_logging2, test_path2, test_method2, test_limit_num2)

    pr.print_the_number_of_data_split_document_size(all_data2)

    # Remove words outside the vocabulary
    test_data2 = pr.remove_words_outside_the_vocabulary(all_data2, vocabulary)
    test_logging2 = deepcopy(all_logging2)
    test_path2 = deepcopy(all_path2)
    test_method2 = deepcopy(all_method2)

#    merge_train_data.append(train_data2)
#   merge_test_data.append(test_data2)
    for layer in model.layers:
        if layer.name == "input_2":
            max_sentence_len2 = layer.get_output_at(0).get_shape().as_list()[1]
            break
#max_sentence_len2 = model.layers[0].get_output_at(0).get_shape().as_list()[1]

    # Make model input.
    X_test2, Y_test2 = pr.create_deep_learning_input_data(test_data2, test_logging2, max_sentence_len2, embed_size_word2vec, vocabulary, wordvec_model)


if input_file_num >= 3:
    pr.data_shuffle(all_data3, 1398)
    pr.data_shuffle(all_logging3, 1398)
    pr.data_shuffle(all_path3, 1398)
    pr.data_shuffle(all_method3, 1398)

    # Set data document size.
# all_data3, all_logging3, all_path3, all_method3 = pr.set_document_size2(standard_data,all_data3,all_logging3, all_path3, all_method3, min_sentence_len, max_sentence_len)

    # Split cross validation set
    '''
    train_data3, test_data3 = pr.split_train_test_data(model_test_rate, all_data3, 10)
    train_logging3, test_logging3 = pr.split_train_test_data(model_test_rate, all_logging3, 10)
    train_path3, test_path3 = pr.split_train_test_data(model_test_rate, all_path3, 10)
    train_method3, test_method3 = pr.split_train_test_data(model_test_rate, all_method3, 10)
    '''
    '''
    zero, one = ev.return_the_number_of_label_data(train_logging3)
    print(input_filename[2])
    print("After set document size of train data, the number of zero and one label data : ",zero, one)
    train_limit_num3 = min(zero, one)
    zero, one = ev.return_the_number_of_label_data(test_logging3)
    print("After set document size of test data, the number of zero and one label data : ",zero, one)
    test_limit_num3 = min(zero, one)
    '''
    zero, one = ev.return_the_number_of_label_data(all_logging3)
    print("The number of zero and one label data : ",zero, one)
    # Balance out the number of data.
#   train_data3, train_logging3, train_path3, train_method3 = pr.balance_out_data(train_data3, train_logging3, train_path3, train_method3, train_limit_num3)
#   test_data3, test_logging3, test_path3, test_method3 = pr.balance_out_data(test_data3, test_logging3, test_path3, test_method3, test_limit_num3)
    pr.print_the_number_of_data_split_document_size(all_data3)

    # Remove words outside the vocabulary
    test_data3 = pr.remove_words_outside_the_vocabulary(all_data3, vocabulary)
    test_logging3 = deepcopy(all_logging3)
    test_path3 = deepcopy(all_path3)
    test_method3 = deepcopy(all_method3)

#    merge_train_data.append(train_data3)
#    merge_test_data.append(test_data3)
    for layer in model.layers:
        if layer.name == "input_3":
            max_sentence_len3 = layer.get_output_at(0).get_shape().as_list()[1]
            break

#    max_sentence_len = model.layer[0].get_output_at(0).get_shape().as_list()[1]

    # Make model input.
    X_test3, Y_test3 = pr.create_deep_learning_input_data(test_data3, test_logging3, max_sentence_len3, embed_size_word2vec, vocabulary, wordvec_model)


if input_file_num >= 4:
    pr.data_shuffle(all_data4, 1398)
    pr.data_shuffle(all_logging4, 1398)
    pr.data_shuffle(all_path4, 1398)
    pr.data_shuffle(all_method4, 1398)

    # Set data document size.
#all_data4, all_logging4, all_path4, all_method4 = pr.set_document_size2(standard_data,all_data4,all_logging4, all_path4, all_method4, min_sentence_len, max_sentence_len)

    # Split cross validation set
    '''
    train_data4, test_data4 = pr.split_train_test_data(model_test_rate, all_data4, 10)
    train_logging4, test_logging4 = pr.split_train_test_data(model_test_rate, all_logging4, 10)
    train_path4, test_path4 = pr.split_train_test_data(model_test_rate, all_path4, 10)
    train_method4, test_method4 = pr.split_train_test_data(model_test_rate, all_method4, 10)
    '''
    '''
    zero, one = ev.return_the_number_of_label_data(train_logging4)
    print(input_filename[3])
    print("After set document size of train data, the number of zero and one label data : ",zero, one)
    train_limit_num4 = min(zero, one)
    zero, one = ev.return_the_number_of_label_data(test_logging4)
    print("After set document size of test data, the number of zero and one label data : ",zero, one)
    test_limit_num4 = min(zero, one)
    '''
    zero, one = ev.return_the_number_of_label_data(all_logging4)
    print("The number of zero and one label data : ",zero, one)
    # Balance out the number of data.
#   train_data4, train_logging4, train_path4, train_method4 = pr.balance_out_data(train_data4, train_logging4, train_path4, train_method4, train_limit_num4)
#   test_data4, test_logging4, test_path4, test_method4 = pr.balance_out_data(test_data4, test_logging4, test_path4, test_method4, test_limit_num4)

    pr.print_the_number_of_data_split_document_size(all_data4)

    # Remove words outside the vocabulary
    test_data4 = pr.remove_words_outside_the_vocabulary(all_data4, vocabulary)
    test_logging4 = deepcopy(all_logging4)
    test_path4 = deepcopy(all_path4)
    test_method4 = deepcopy(all_method4)

#    merge_train_data.append(train_data4)
#    merge_test_data.append(test_data4)
    for layer in model.layers:
        if layer.name == "input_4":
            max_sentence_len4 = layer.get_output_at(0).get_shape().as_list()[1]
            break

#    max_sentence_len4 = model.layers[0].get_output_at(0).get_shape().as_list()[1]

    # Make model input.
    X_test4, Y_test4 = pr.create_deep_learning_input_data(test_data4, test_logging4, max_sentence_len4, embed_size_word2vec, vocabulary, wordvec_model)
'''
# Save input data to json format.
pr.make_input_data_json2(filename, model_test_rate, merge_train_data, train_logging, train_path, train_method, merge_test_data, test_logging, test_path, test_method)
'''
print("Count model parameter.")
model.count_params()
print("Get a short summary of each layer dimensions and parameters.")
model.summary()

if input_file_num == 1:
    predict = model.predict(X_test)
elif input_file_num == 2:
    predict = model.predict([X_test, X_test2])
elif input_file_num == 3:
    predict = model.predict([X_test, X_test2, X_test3])
elif input_file_num == 4:
    predict = model.predict([X_test, X_test2, X_test3, X_test4])

predict_1d = predict.argmax(axis = 1)
Y_test_1d = Y_test.argmax(axis = 1)
preds = predict[:,1]

accuracy = ev.accuracy(predict_1d, Y_test_1d)

test_zero, test_one = ev.return_the_number_of_label_data(Y_test_1d)
choose_zero, choose_one = ev.return_the_number_of_predict_data(predict_1d)

# Print evaluate results.
print("\nTest accuracy:", accuracy)
print("\ntest data size : ", test_zero+test_one)
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
'''
# Save predict rate with path and method of data.
TP_file_name = "Code_snippet/end2/"+str(model_test_rate)+"/TP_"+filename+"_ep"+str(ep+1)+"_"+str(count)+".txt"
FP_file_name = "Code_snippet/end2/"+str(model_test_rate)+"/FP_"+filename+"_ep"+str(ep+1)+"_"+str(count)+ ".txt"
TN_file_name = "Code_snippet/end2/"+str(model_test_rate)+"/TN_"+filename+"_ep"+str(ep+1)+"_"+str(count)+ ".txt"
FN_file_name = "Code_snippet/end2/"+str(model_test_rate)+"/FN_"+filename+"_ep"+str(ep+1)+"_"+str(count)+ ".txt"
ev.save_predict_rate(TP_file_name, FP_file_name, TN_file_name, FN_file_name, predict_1d, predict, Y_test_1d, test_path, test_method)

# Save ROC graph.
roc_name = "ROC/end2/"+str(model_test_rate) +"/JAL_"+filename+"_ep"+str(ep+1)+"_"+str(count)
ev.save_ROC(roc_count, Y_test_1d, preds, roc_name)
'''
del model


#input 형태 1가지 일떄만 가능. 2가지 이상인 경우 미구현 상태.
if pred_by_sentence_len:
    print(np.shape(X_test))
    X_test_len = [[] for i in range(41)]
    Y_test_len = [[] for i in range(41)]

    for i, item in enumerate(st_X_test):
        sentence_len = 0
        for j, item2 in enumerate(item):
            if np.count_nonzero(item2) == 0:
                pass
            else:
                sentence_len += 1
        if sentence_len > 400:
            X_test_len[40].append(X_test[i].tolist())
            Y_test_len[40].append(Y_test[i])
        X_test_len[sentence_len//10].append(X_test[i].tolist())
        Y_test_len[sentence_len//10].append(Y_test[i])

#X_test_len = np.array(X_test_len)
    Y_test_len = np.array(Y_test_len)
    for i in range(41):
        print('\n'+str((i)*10)+'~'+str((i+1)*10))
        if input_file_num == 1:
            if len(X_test_len[i]) == 0:
                continue
            temp = deepcopy(X_test_len[i])
            temp = np.array(temp)
            predict = model.predict(temp)
        elif input_file_num == 2:
            if len(X_test_len[i]) == 0:
                continue
            temp = deepcopy(X_test_len[i])
            temp = np.array(temp)
            predict = model.predict([X_test, X_test2])
        elif input_file_num == 3:
            if len(X_test_len[i]) == 0:
                continue
            predict = model.predict([X_test, X_test2, X_test3])
        elif input_file_num == 4:
            if len(X_test_len[i]) == 0:
                continue
            predict = model.predict([X_test, X_test2, X_test3, X_test4])

        temp_Y = deepcopy(Y_test_len[i])
        temp_Y = np.array(temp_Y)
        predict_1d = predict.argmax(axis = 1)
        Y_test_1d = temp_Y.argmax(axis = 1)
#Y_test_1d = Y_test_len[i].argmax(axis = 1)
        preds = predict[:,1]

        accuracy = ev.accuracy(predict_1d, Y_test_1d)

        test_zero, test_one = ev.return_the_number_of_label_data(Y_test_1d)
        choose_zero, choose_one = ev.return_the_number_of_predict_data(predict_1d)

# Print evaluate results.
        print("\nTest accuracy:", accuracy)
        print("\ntest data size : ", test_zero+test_one)
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
        '''
# Save predict rate with path and method of data.
        TP_file_name = "Code_snippet/end2/"+str(model_test_rate)+"/TP_"+filename+"_ep"+str(ep+1)+"_"+str(count)+".txt"
        FP_file_name = "Code_snippet/end2/"+str(model_test_rate)+"/FP_"+filename+"_ep"+str(ep+1)+"_"+str(count)+ ".txt"
        TN_file_name = "Code_snippet/end2/"+str(model_test_rate)+"/TN_"+filename+"_ep"+str(ep+1)+"_"+str(count)+ ".txt"
        FN_file_name = "Code_snippet/end2/"+str(model_test_rate)+"/FN_"+filename+"_ep"+str(ep+1)+"_"+str(count)+ ".txt"
        ev.save_predict_rate(TP_file_name, FP_file_name, TN_file_name, FN_file_name, predict_1d, predict, Y_test_1d, test_path, test_method)

# Save ROC graph.
        roc_name = "ROC/end2/"+str(model_test_rate) +"/JAL_"+filename+"_ep"+str(ep+1)+"_"+str(count)
        ev.save_ROC(roc_count, Y_test_1d, preds, roc_name)
        '''
    del model
