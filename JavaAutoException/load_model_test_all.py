# Required dependencies
# 1. NLTK
# 2. Gensim for word2vec
# 3. Keras with tensorflow/theano backend
import preprocessing_data as pr
import model as md
import evaluation as ev

import sys
from copy import deepcopy
from keras.models import model_from_json

input_file_num = int(sys.argv[4])
#========================================================================================
# Initializing Hyper parameter
#========================================================================================
#1. Word2vec parameters
min_word_frequency_word2vec = int(sys.argv[input_file_num+6])
embed_size_word2vec = 200

#2. Classifier hyperparameters
min_sentence_len = int(sys.argv[input_file_num+7])
max_sentence_len = int(sys.argv[input_file_num+8])
model_test_rate = int(sys.argv[input_file_num+5])
model_count = 1
label_name = "isException"

#3. Input file name
target_file = sys.argv[1]
source_file = sys.argv[2]
input_filename = []
for i in range(input_file_num):
    input_filename.append(sys.argv[i+5])
    input_filename[i] = source_file + '-' + input_filename[i]
target_type = sys.argv[3]
target_filename = target_file + '-' + target_type
java_auto_exception_json = deepcopy(input_filename)
for i in range(input_file_num):
    java_auto_exception_json[i] = "../MakeJSON/output/Exception/real_last/"+input_filename[i]+".json"
wordvec_model_name = target_filename + '_frequency'+str(min_word_frequency_word2vec)
model_filename = target_filename + "_sentence_balanced_min"+str(min_sentence_len)+"_max"+str(max_sentence_len)+"_frequency"+str(min_word_frequency_word2vec)

#========================================================================================
# Preprocess the java auto exception, extract the vocabulary and learn the word2vec representation
#========================================================================================
if input_file_num >= 1:
    all_data, all_exception, all_path, all_method = pr.open_data(java_auto_exception_json[0], label_name)

    zero, one = ev.return_the_number_of_label_data(all_exception)
    print("First data")
    print("zero : ", zero)
    print("one : ", one)
    print()

if input_file_num >= 2:
    all_data2, all_exception2, all_path2, all_method2 = pr.open_data(java_auto_exception_json[1], label_name)

    zero, one = ev.return_the_number_of_label_data(all_exception2)
    print("Second data")
    print("zero : ", zero)
    print("one : ", one)
    print()

if input_file_num >= 3:
    all_data3, all_exception3, all_path3, all_method3 = pr.open_data(java_auto_exception_json[2], label_name)

    zero, one = ev.return_the_number_of_label_data(all_exception3)
    print("Third data")
    print("zero : ", zero)
    print("one : ", one)
    print()

if input_file_num >= 4:
    all_data4, all_exception4, all_path4, all_method4 = pr.open_data(java_auto_exception_json[3], label_name)

    zero, one = ev.return_the_number_of_label_data(all_exception4)
    print("4th data")
    print("zero : ", zero)
    print("one : ", one)
    print()


# Load the word2vec model and vocabulary
model_name = "Wordvec_Model/real_last/"+wordvec_model_name+"_"+str(model_count)+".model"
wordvec_model, vocabulary = pr.load_word2vec_model(model_name)

# Load model
model_json = "Model/real_last/"+str(model_test_rate)+"/best/JAE_"+model_filename+"_"+str(model_count)+"_model.json"
model_h5 = "Model/real_last/"+str(model_test_rate)+"/best/JAE_"+model_filename+"_"+str(model_count)+"_model.h5"
model = md.load_model(model_json, model_h5)
'''
json_file = open(model_json, "r")
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights(model_h5)
print("Loaded model from disk")
'''
print("Count model parameter.")
model.count_params()
print("Get a short summary of each layer dimensions and parameters.")
model.summary()

#========================================================================================
# Split cross validation sets and perform deep learning + softmax based classification
#========================================================================================

merge_train_data = []
merge_test_data = []
if input_file_num >= 1:
    pr.data_shuffle(all_data, 1398)
    pr.data_shuffle(all_exception, 1398)
    pr.data_shuffle(all_path, 1398)
    pr.data_shuffle(all_method, 1398)
  
    zero, one = ev.return_the_number_of_label_data(all_exception)
    print("The number of zero and one label data : ",zero, one)
    pr.print_the_number_of_data_split_document_size(all_data)

    # Remove words outside the vocabulary
    test_data = pr.remove_words_outside_the_vocabulary(all_data, vocabulary)
    test_exception = deepcopy(all_exception)

#    merge_train_data.append(train_data)
#    merge_test_data.append(test_data)

    for layer in model.layers:
        if layer.name == "input_1":
            max_sentence_len = layer.get_output_at(0).get_shape().as_list()[1]
            break
    print(max_sentence_len)

    # Make model input.
    X_test, Y_test = pr.create_deep_learning_input_data(test_data, test_exception, max_sentence_len, embed_size_word2vec, vocabulary, wordvec_model)

if input_file_num >= 2:
    pr.data_shuffle(all_data2, 1398)
    pr.data_shuffle(all_exception2, 1398)
    pr.data_shuffle(all_path2, 1398)
    pr.data_shuffle(all_method2, 1398)

    zero, one = ev.return_the_number_of_label_data(all_exception2)
    print("The number of zero and one label data : ",zero, one)
    pr.print_the_number_of_data_split_document_size(all_data2)

    # Remove words outside the vocabulary
    test_data2 = pr.remove_words_outside_the_vocabulary(all_data2, vocabulary)
    test_exception2 = deepcopy(all_exception2)

    #merge_train_data.append(train_data2)
    #merge_test_data.append(test_data2)

    for layer in model.layers:
        if layer.name == "input_2":
            max_sentence_len2 = layer.get_output_at(0).get_shape().as_list()[1]
            break
    print(max_sentence_len2)

    # Make model input.
    X_test2, Y_test2 = pr.create_deep_learning_input_data(test_data2, test_exception2, max_sentence_len2, embed_size_word2vec, vocabulary, wordvec_model)

if input_file_num >= 3:
    pr.data_shuffle(all_data3, 1398)
    pr.data_shuffle(all_exception3, 1398)
    pr.data_shuffle(all_path3, 1398)
    pr.data_shuffle(all_method3, 1398)

    zero, one = ev.return_the_number_of_label_data(all_exception3)
    print("The number of zero and one label data : ",zero, one)
    pr.print_the_number_of_data_split_document_size(all_data3)

    # Remove words outside the vocabulary
    test_data3 = pr.remove_words_outside_the_vocabulary(all_data3, vocabulary)
    test_exception3 = deepcopy(all_exception3)

   # merge_train_data.append(train_data3)
   # merge_test_data.append(test_data3)

    for layer in model.layers:
        if layer.name == "input_3":
            max_sentence_len3 = layer.get_output_at(0).get_shape().as_list()[1]
            break
    print(max_sentence_len3)

    # Make model input.
    X_test3, Y_test3 = pr.create_deep_learning_input_data(test_data3, test_exception3, max_sentence_len3, embed_size_word2vec, vocabulary, wordvec_model)


if input_file_num >= 4:
    pr.data_shuffle(all_data4, 1398)
    pr.data_shuffle(all_exception4, 1398)
    pr.data_shuffle(all_path4, 1398)
    pr.data_shuffle(all_method4, 1398)

    zero, one = ev.return_the_number_of_label_data(all_exception4)
    print("The number of zero and one label data : ",zero, one)
    pr.print_the_number_of_data_split_document_size(all_data4)

    # Remove words outside the vocabulary
    test_data4 = pr.remove_words_outside_the_vocabulary(all_data4, vocabulary)
    test_exception4 = deepcopy(all_exception4)

    #merge_train_data.append(train_data4)
    #merge_test_data.append(test_data4)

    for layer in model.layers:
        if layer.name == "input_4":
            max_sentence_len4 = layer.get_output_at(0).get_shape().as_list()[1]
            break
    print(max_sentence_len4)

    # Make model input.
    X_test4, Y_test4 = pr.create_deep_learning_input_data(test_data4, test_exception4, max_sentence_len4, embed_size_word2vec, vocabulary, wordvec_model)
'''
# Save input data to json format.
pr.make_input_data_json2(filename, model_test_rate, merge_train_data, train_exception, train_path, train_method, merge_test_data, test_exception, test_path, test_method)
'''

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
TP_file_name = "Code_snippet/real_last/"+str(model_test_rate)+"/TP_"+filename+"_ep"+str(ep+1)+"_"+str(count)+".txt"
FP_file_name = "Code_snippet/real_last/"+str(model_test_rate)+"/FP_"+filename+"_ep"+str(ep+1)+"_"+str(count)+ ".txt"
TN_file_name = "Code_snippet/real_last/"+str(model_test_rate)+"/TN_"+filename+"_ep"+str(ep+1)+"_"+str(count)+ ".txt"
FN_file_name = "Code_snippet/real_last/"+str(model_test_rate)+"/FN_"+filename+"_ep"+str(ep+1)+"_"+str(count)+ ".txt"
ev.save_predict_rate(TP_file_name, FP_file_name, TN_file_name, FN_file_name, predict_1d, predict, Y_test_1d, test_path, test_method)

# Save ROC graph.
roc_name = "ROC/real_last/"+str(model_test_rate) +"/JAE_"+filename+"_ep"+str(ep+1)+"_"+str(count)
ev.save_ROC(roc_count, Y_test_1d, preds, roc_name)
'''
del model
