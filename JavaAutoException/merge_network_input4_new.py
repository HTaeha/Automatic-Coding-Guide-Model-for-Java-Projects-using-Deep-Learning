# Required dependencies
# 1. NLTK
# 2. Gensim for word2vec
# 3. Keras with tensorflow/theano backend
import preprocessing_data as pr
import model as md
import evaluation as ev

import sys
from copy import deepcopy

input_file_num = int(sys.argv[3])
#========================================================================================
# Initializing Hyper parameter
#========================================================================================
#1. Word2vec parameters
min_word_frequency_word2vec = int(sys.argv[input_file_num+5])
embed_size_word2vec = 200
context_window_word2vec = 5

#2. Classifier hyperparameters
numCV = 1
st_min_sentence_len = int(sys.argv[input_file_num+7])
st_max_sentence_len = int(sys.argv[input_file_num+8])
batch_size = 64
LSTM_output_size = 64
epoch_len = 15
test_rate = int(sys.argv[input_file_num+4])
count = 1
model_count = 1
label_name = "isException"

#3. Input file name
input_file = sys.argv[1]
input_filename = []
for i in range(input_file_num):
    input_filename.append(sys.argv[i+4])
    input_filename[i] = input_file + '-' + input_filename[i]
standard_input_type = sys.argv[2]
if input_file_num == 1:
    filename = input_filename[0] + "_min"+str(st_min_sentence_len)+"_max"+str(st_max_sentence_len)+"_frequency"+str(min_word_frequency_word2vec)
    wordvec_model_name = input_filename[0] + "_frequency"+str(min_word_frequency_word2vec)
else:
    filename = input_file + "-CAST_m_"
    for i in range(input_file_num):
        filename = filename + input_filename[i][len(input_file)+1]
    filename = filename + "_min"+str(st_min_sentence_len)+'_max'+str(st_max_sentence_len)+'_frequency'+str(min_word_frequency_word2vec)
    wordvec_model_name = input_file + "-CAST_m_"
    for i in range(input_file_num):
        wordvec_model_name = wordvec_model_name + input_filename[i][len(input_file)+1]
    wordvec_model_name = wordvec_model_name + '_frequency'+str(min_word_frequency_word2vec)
standard_input_filename = input_file + '-' + standard_input_type
java_auto_exception_json = deepcopy(input_filename)
for i in range(input_file_num):
    java_auto_exception_json[i] = "../MakeJSON/output/Exception/real_last/"+input_filename[i]+".json"
standard_java_auto_exception_json = '../MakeJSON/output/Exception/real_last/'+standard_input_filename+'.json'
standard_wordvec_model_name = standard_input_filename + '_frequency'+str(min_word_frequency_word2vec)

#========================================================================================
# Preprocess the java auto exception, extract the vocabulary and learn the word2vec representation
#========================================================================================

standard_data, standard_exception, standard_path, standard_method = pr.open_data(standard_java_auto_exception_json, label_name)

zero, one = ev.return_the_number_of_label_data(standard_exception)
print("Standard data")
print("zero : ", zero)
print("one : ", one)
print()

if input_file_num >= 1:
    all_data, all_exception, all_path, all_method = pr.open_data(java_auto_exception_json[0], label_name)

    zero, one = ev.return_the_number_of_label_data(all_exception)
    print("First data")
    print("zero : ", zero)
    print("one : ", one)
    print()
    merge_data = deepcopy(all_data)

if input_file_num >= 2:
    all_data2, all_exception2, all_path2, all_method2 = pr.open_data(java_auto_exception_json[1], label_name)

    zero, one = ev.return_the_number_of_label_data(all_exception2)
    print("Second data")
    print("zero : ", zero)
    print("one : ", one)
    print()
    merge_data += all_data2

if input_file_num >= 3:
    all_data3, all_exception3, all_path3, all_method3 = pr.open_data(java_auto_exception_json[2], label_name)

    zero, one = ev.return_the_number_of_label_data(all_exception3)
    print("Third data")
    print("zero : ", zero)
    print("one : ", one)
    print()
    merge_data += all_data3

if input_file_num >= 4:
    all_data4, all_exception4, all_path4, all_method4 = pr.open_data(java_auto_exception_json[3], label_name)

    zero, one = ev.return_the_number_of_label_data(all_exception4)
    print("4th data")
    print("zero : ", zero)
    print("one : ", one)
    print()
    merge_data += all_data4

'''
for i, item in enumerate(all_data):
    if all_exception[i] != all_exception3[i]:
#    if all_path[i] != all_path3[i]:# or all_path2[i] != all_path3[i]:
#       if all_method[i] != all_method3[i]:# or all_method2[i] != all_method3[i]:
        print(all_data[i], all_path[i], all_method[i])
        print(all_data3[i], all_path3[i], all_method3[i])
'''
# Learn the word2vec model and extract vocabulary
model_name = "Wordvec_Model/real_last/"+wordvec_model_name+"_"+str(count)+".model"
standard_model_name = "Wordvec_Model/real_last/"+standard_wordvec_model_name+"_"+str(count)+".model"
first_execution = int(sys.argv[input_file_num+6])
if first_execution:
    wordvec_model, vocabulary = pr.train_word2vec_model(merge_data, min_word_frequency_word2vec, embed_size_word2vec, context_window_word2vec, model_name)
else:
    wordvec_model, vocabulary = pr.load_word2vec_model(model_name)
sys.exit(1)
#========================================================================================
# Split cross validation sets and perform deep learning + softmax based classification
#========================================================================================
pr.data_shuffle(standard_data, 1398)

merge_train_data = []
merge_test_data = []
if input_file_num >= 1:
    pr.data_shuffle(all_data, 1398)
    pr.data_shuffle(all_exception, 1398)
    pr.data_shuffle(all_path, 1398)
    pr.data_shuffle(all_method, 1398)
   
    # Set data document size.
    all_data, all_exception, all_path, all_method = pr.set_document_size2(standard_data, all_data, all_exception, all_path, all_method, st_min_sentence_len, st_max_sentence_len)

    # Split cross validation set
    train_data, test_data = pr.split_train_test_data(test_rate, all_data, 10)
    train_exception, test_exception = pr.split_train_test_data(test_rate, all_exception, 10)
    train_path, test_path = pr.split_train_test_data(test_rate, all_path, 10)
    train_method, test_method = pr.split_train_test_data(test_rate, all_method, 10)

    zero, one = ev.return_the_number_of_label_data(train_exception)
    print(input_filename[0])
    print("After set document size of train data, the number of zero and one label data : ",zero, one)
    train_limit_num = min(zero, one)
    zero, one = ev.return_the_number_of_label_data(test_exception)
    print("After set document size of test data, the number of zero and one label data : ",zero, one)
    test_limit_num = min(zero, one)

    pr.print_the_number_of_data_split_document_size(train_data+test_data)
    # Balance out the number of data.
    train_data, train_exception, train_path, train_method = pr.balance_out_data(train_data, train_exception, train_path, train_method, train_limit_num)
    test_data, test_exception, test_path, test_method = pr.balance_out_data(test_data, test_exception, test_path, test_method, test_limit_num)

    print("After balance out data.")
    print(input_filename[0])
    pr.print_the_number_of_data_split_document_size(train_data+test_data)

    # Remove words outside the vocabulary
    train_data = pr.remove_words_outside_the_vocabulary(train_data, vocabulary)
    test_data = pr.remove_words_outside_the_vocabulary(test_data, vocabulary)

    merge_train_data.append(train_data)
    merge_test_data.append(test_data)

    max_sentence_len = pr.check_max_sentence_len(train_data + test_data, train_path+test_path, train_method+test_method)
    print(max_sentence_len)

    # Make model input.
    X_train, Y_train = pr.create_deep_learning_input_data(train_data, train_exception, max_sentence_len, embed_size_word2vec, vocabulary, wordvec_model)
    X_test, Y_test = pr.create_deep_learning_input_data(test_data, test_exception, max_sentence_len, embed_size_word2vec, vocabulary, wordvec_model)

if input_file_num >= 2:
    pr.data_shuffle(all_data2, 1398)
    pr.data_shuffle(all_exception2, 1398)
    pr.data_shuffle(all_path2, 1398)
    pr.data_shuffle(all_method2, 1398)
    
    # Set data document size.
    all_data2, all_exception2, all_path2, all_method2 = pr.set_document_size2(standard_data,all_data2,all_exception2, all_path2, all_method2, st_min_sentence_len, st_max_sentence_len)

    # Split cross validation set
    train_data2, test_data2 = pr.split_train_test_data(test_rate, all_data2, 10)
    train_exception2, test_exception2 = pr.split_train_test_data(test_rate, all_exception2, 10)
    train_path2, test_path2 = pr.split_train_test_data(test_rate, all_path2, 10)
    train_method2, test_method2 = pr.split_train_test_data(test_rate, all_method2, 10)

    zero, one = ev.return_the_number_of_label_data(train_exception2)
    print(input_filename[1])
    print("After set document size of train data, the number of zero and one label data : ",zero, one)
    train_limit_num2 = min(zero, one)
    zero, one = ev.return_the_number_of_label_data(test_exception2)
    print("After set document size of test data, the number of zero and one label data : ",zero, one)
    test_limit_num2 = min(zero, one)

    # Balance out the number of data.
    train_data2, train_exception2, train_path2, train_method2 = pr.balance_out_data(train_data2, train_exception2, train_path2, train_method2, train_limit_num2)
    test_data2, test_exception2, test_path2, test_method2 = pr.balance_out_data(test_data2, test_exception2, test_path2, test_method2, test_limit_num2)

    print("After balance out data.")
    print(input_filename[1])
    pr.print_the_number_of_data_split_document_size(train_data2+test_data2)

    # Remove words outside the vocabulary
    train_data2 = pr.remove_words_outside_the_vocabulary(train_data2, vocabulary)
    test_data2 = pr.remove_words_outside_the_vocabulary(test_data2, vocabulary)

    merge_train_data.append(train_data2)
    merge_test_data.append(test_data2)

    max_sentence_len2 = pr.check_max_sentence_len(train_data2 + test_data2, train_path2+test_path2, train_method2+test_method2)

    print(max_sentence_len2)
    # Make model input.
    X_train2, Y_train2 = pr.create_deep_learning_input_data(train_data2, train_exception2, max_sentence_len2, embed_size_word2vec, vocabulary, wordvec_model)
    X_test2, Y_test2 = pr.create_deep_learning_input_data(test_data2, test_exception2, max_sentence_len2, embed_size_word2vec, vocabulary, wordvec_model)


if input_file_num >= 3:
    pr.data_shuffle(all_data3, 1398)
    pr.data_shuffle(all_exception3, 1398)
    pr.data_shuffle(all_path3, 1398)
    pr.data_shuffle(all_method3, 1398)

    # Set data document size.
    all_data3, all_exception3, all_path3, all_method3 = pr.set_document_size2(standard_data,all_data3,all_exception3, all_path3, all_method3, st_min_sentence_len, st_max_sentence_len)

    # Split cross validation set
    train_data3, test_data3 = pr.split_train_test_data(test_rate, all_data3, 10)
    train_exception3, test_exception3 = pr.split_train_test_data(test_rate, all_exception3, 10)
    train_path3, test_path3 = pr.split_train_test_data(test_rate, all_path3, 10)
    train_method3, test_method3 = pr.split_train_test_data(test_rate, all_method3, 10)

    zero, one = ev.return_the_number_of_label_data(train_exception3)
    print(input_filename[2])
    print("After set document size of train data, the number of zero and one label data : ",zero, one)
    train_limit_num3 = min(zero, one)
    zero, one = ev.return_the_number_of_label_data(test_exception3)
    print("After set document size of test data, the number of zero and one label data : ",zero, one)
    test_limit_num3 = min(zero, one)

    # Balance out the number of data.
    train_data3, train_exception3, train_path3, train_method3 = pr.balance_out_data(train_data3, train_exception3, train_path3, train_method3, train_limit_num3)
    test_data3, test_exception3, test_path3, test_method3 = pr.balance_out_data(test_data3, test_exception3, test_path3, test_method3, test_limit_num3)

    print("After balance out data.")
    print(input_filename[2])
    pr.print_the_number_of_data_split_document_size(train_data3+test_data3)

    # Remove words outside the vocabulary
    train_data3 = pr.remove_words_outside_the_vocabulary(train_data3, vocabulary)
    test_data3 = pr.remove_words_outside_the_vocabulary(test_data3, vocabulary)

    merge_train_data.append(train_data3)
    merge_test_data.append(test_data3)

    max_sentence_len3 = pr.check_max_sentence_len(train_data3 + test_data3, train_path3+test_path3, train_method3+test_method3)

    print(max_sentence_len3)
    # Make model input.
    X_train3, Y_train3 = pr.create_deep_learning_input_data(train_data3, train_exception3, max_sentence_len3, embed_size_word2vec, vocabulary, wordvec_model)
    X_test3, Y_test3 = pr.create_deep_learning_input_data(test_data3, test_exception3, max_sentence_len3, embed_size_word2vec, vocabulary, wordvec_model)


if input_file_num >= 4:
    pr.data_shuffle(all_data4, 1398)
    pr.data_shuffle(all_exception4, 1398)
    pr.data_shuffle(all_path4, 1398)
    pr.data_shuffle(all_method4, 1398)

    # Set data document size.
    all_data4, all_exception4, all_path4, all_method4 = pr.set_document_size2(standard_data,all_data4,all_exception4, all_path4, all_method4, st_min_sentence_len, st_max_sentence_len)

    # Split cross validation set
    train_data4, test_data4 = pr.split_train_test_data(test_rate, all_data4, 10)
    train_exception4, test_exception4 = pr.split_train_test_data(test_rate, all_exception4, 10)
    train_path4, test_path4 = pr.split_train_test_data(test_rate, all_path4, 10)
    train_method4, test_method4 = pr.split_train_test_data(test_rate, all_method4, 10)

    zero, one = ev.return_the_number_of_label_data(train_exception4)
    print(input_filename[3])
    print("After set document size of train data, the number of zero and one label data : ",zero, one)
    train_limit_num4 = min(zero, one)
    zero, one = ev.return_the_number_of_label_data(test_exception4)
    print("After set document size of test data, the number of zero and one label data : ",zero, one)
    test_limit_num4 = min(zero, one)

    # Balance out the number of data.
    train_data4, train_exception4, train_path4, train_method4 = pr.balance_out_data(train_data4, train_exception4, train_path4, train_method4, train_limit_num4)
    test_data4, test_exception4, test_path4, test_method4 = pr.balance_out_data(test_data4, test_exception4, test_path4, test_method4, test_limit_num4)

    print("After balance out data.")
    print(input_filename[3])
    pr.print_the_number_of_data_split_document_size(train_data4+test_data4)

    # Remove words outside the vocabulary
    train_data4 = pr.remove_words_outside_the_vocabulary(train_data4, vocabulary)
    test_data4 = pr.remove_words_outside_the_vocabulary(test_data4, vocabulary)

    merge_train_data.append(train_data4)
    merge_test_data.append(test_data4)

    max_sentence_len4 = pr.check_max_sentence_len(train_data4 + test_data4, train_path4+test_path4, train_method4+test_method4)

    print(max_sentence_len4)
    # Make model input.
    X_train4, Y_train4 = pr.create_deep_learning_input_data(train_data4, train_exception4, max_sentence_len4, embed_size_word2vec, vocabulary, wordvec_model)
    X_test4, Y_test4 = pr.create_deep_learning_input_data(test_data4, test_exception4, max_sentence_len4, embed_size_word2vec, vocabulary, wordvec_model)

sys.exit(1)
# Save input data to json format.
input_train_data_path = "input_data/real_last/train/"+str(test_rate)+"/"+filename+".json"
input_test_data_path = "input_data/real_last/test/"+str(test_rate)+"/"+filename+".json"
pr.make_input_merge_data_json(input_train_data_path, label_name, merge_train_data, train_exception, train_path, train_method)
pr.make_input_merge_data_json(input_test_data_path, label_name, merge_test_data, test_exception, test_path, test_method)

# Construct the deep learning model
if input_file_num == 1:
    model = md.bidirectional_RNN(max_sentence_len, embed_size_word2vec, LSTM_output_size)
elif input_file_num == 2:
    model = md.merge_2_bidirectional_RNN(max_sentence_len, max_sentence_len2, embed_size_word2vec, LSTM_output_size)
elif input_file_num == 3:
    model = md.merge_3_bidirectional_RNN(max_sentence_len, max_sentence_len2, max_sentence_len3,  embed_size_word2vec, LSTM_output_size)
elif input_file_num == 4:
    model = md.merge_4_bidirectional_RNN(max_sentence_len, max_sentence_len2, max_sentence_len3, max_sentence_len4, embed_size_word2vec, LSTM_output_size)
print("Count model parameter.")
model.count_params()
print("Get a short summary of each layer dimensions and parameters.")
model.summary()

accuracy_list = []
roc_count = 1
for ep in range(epoch_len) :
    print(str(ep+1) + "\n")
    if input_file_num == 1:
        hist = model.fit(X_train, Y_train, batch_size = batch_size, epochs = 1, verbose = 1)
        predict = model.predict(X_test)
    elif input_file_num == 2:
        hist = model.fit([X_train, X_train2], Y_train, batch_size = batch_size, epochs=1, verbose=1)
        predict = model.predict([X_test, X_test2])
    elif input_file_num == 3:
        hist = model.fit([X_train, X_train2, X_train3], Y_train, batch_size = batch_size, epochs=1, verbose=1)
        predict = model.predict([X_test, X_test2, X_test3])
    elif input_file_num == 4:
        hist = model.fit([X_train, X_train2, X_train3, X_train4], Y_train, batch_size = batch_size, epochs=1, verbose=1)
        predict = model.predict([X_test, X_test2, X_test3, X_test4])

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

    # Save predict rate with path and method of data.
    TP_file_name = "Code_snippet/real_last/"+str(test_rate)+"/TP_"+filename+"_ep"+str(ep+1)+"_"+str(count)+".txt"
    FP_file_name = "Code_snippet/real_last/"+str(test_rate)+"/FP_"+filename+"_ep"+str(ep+1)+"_"+str(count)+ ".txt"
    TN_file_name = "Code_snippet/real_last/"+str(test_rate)+"/TN_"+filename+"_ep"+str(ep+1)+"_"+str(count)+ ".txt"
    FN_file_name = "Code_snippet/real_last/"+str(test_rate)+"/FN_"+filename+"_ep"+str(ep+1)+"_"+str(count)+ ".txt"
    ev.save_predict_rate(TP_file_name, FP_file_name, TN_file_name, FN_file_name, predict_1d, predict, Y_test_1d, test_path, test_method)
    # Save ROC graph.
    roc_name = "ROC/real_last/"+str(test_rate) +"/JAE_"+filename+"_ep"+str(ep+1)+"_"+str(count)
    ev.save_ROC(roc_count, Y_test_1d, preds, roc_name)
    roc_count += 1

    # Save model.
    model_name = "Model/real_last/"+str(test_rate) + "/JAE_"+filename+"_ep"+str(ep+1)+"_"+str(count)+"_model.json"
    weight_name = "Model/real_last/"+str(test_rate) + "/JAE_"+filename+"_ep"+str(ep+1)+"_"+str(count)+"_model.h5"
    md.save_model(model, model_name, weight_name)

print(accuracy_list)
print("max accuracy : ", max(accuracy_list))
del model
