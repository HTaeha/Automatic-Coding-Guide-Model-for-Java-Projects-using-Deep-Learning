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
label_name = "isLogged"

#3. Input file name
input_file = sys.argv[1]
input_filename = []
for i in range(input_file_num):
    input_filename.append(sys.argv[i+4])
    input_filename[i] = input_file + '-' + input_filename[i]
standard_input_type = sys.argv[2]
if input_file_num == 1:
    wordvec_model_name = input_filename[0] + "_frequency"+str(min_word_frequency_word2vec)
else:
    wordvec_model_name = input_file + "-CAST_m_"
    for i in range(input_file_num):
        wordvec_model_name = wordvec_model_name + input_filename[i][len(input_file)+1]
    wordvec_model_name = wordvec_model_name + '_frequency'+str(min_word_frequency_word2vec)
standard_input_filename = input_file + '-' + standard_input_type
java_auto_logging_json = deepcopy(input_filename)
for i in range(input_file_num):
    java_auto_logging_json[i] = "../MakeJSON/output/Logging/end2/"+input_filename[i]+".json"
standard_java_auto_logging_json = '../MakeJSON/output/Logging/end2/'+standard_input_filename+'.json'
standard_wordvec_model_name = standard_input_filename + '_frequency'+str(min_word_frequency_word2vec)

#========================================================================================
# Preprocess the java auto logging, extract the vocabulary and learn the word2vec representation
#========================================================================================

standard_data, standard_logging, standard_path, standard_method = pr.open_data(standard_java_auto_logging_json, label_name)

zero, one = ev.return_the_number_of_label_data(standard_logging)
print("Standard data")
print("zero : ", zero)
print("one : ", one)
print()

if input_file_num >= 1:
    all_data, all_logging, all_path, all_method = pr.open_data(java_auto_logging_json[0], label_name)

    zero, one = ev.return_the_number_of_label_data(all_logging)
    print("First data")
    print("zero : ", zero)
    print("one : ", one)
    print()
    merge_data = deepcopy(all_data)

if input_file_num >= 2:
    all_data1, all_logging2, all_path2, all_method2 = pr.open_data(java_auto_logging_json[1], label_name)

    zero, one = ev.return_the_number_of_label_data(all_logging2)
    print("Second data")
    print("zero : ", zero)
    print("one : ", one)
    print()
    merge_data += all_data1

if input_file_num >= 3:
    all_data3, all_logging3, all_path3, all_method3 = pr.open_data(java_auto_logging_json[2], label_name)

    zero, one = ev.return_the_number_of_label_data(all_logging3)
    print("Third data")
    print("zero : ", zero)
    print("one : ", one)
    print()
    merge_data += all_data3

if input_file_num >= 4:
    all_data4, all_logging4, all_path4, all_method4 = pr.open_data(java_auto_logging_json[3], label_name)

    zero, one = ev.return_the_number_of_label_data(all_logging4)
    print("4th data")
    print("zero : ", zero)
    print("one : ", one)
    print()
    merge_data += all_data4

for i, item in enumerate(all_data):
    if all_path[i] != all_path2[i]:# or all_path2[i] != all_path3[i]:
        if all_method[i] != all_method2[i]:# or all_method2[i] != all_method3[i]:
            print(all_data[i], all_path[i], all_method[i])
            print(all_data1[i], all_path2[i], all_method2[i])
# Learn the word2vec model and extract vocabulary
model_name = "Wordvec_Model/end2/"+wordvec_model_name+"_"+str(count)+".model"
standard_model_name = "Wordvec_Model/end2/"+standard_wordvec_model_name+"_"+str(count)+".model"
first_execution = int(sys.argv[input_file_num+6])
if first_execution:
    wordvec_model, vocabulary = pr.train_word2vec_model(merge_data, min_word_frequency_word2vec, embed_size_word2vec, context_window_word2vec, model_name)
    standard_wordvec_model, standard_vocabulary = pr.train_word2vec_model(standard_data, min_word_frequency_word2vec, embed_size_word2vec, context_window_word2vec, standard_model_name)
else:
    wordvec_model, vocabulary = pr.load_word2vec_model(model_name)
    standard_wordvec_model, standard_vocabulary = pr.load_word2vec_model(standard_model_name)

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
   
    all_data = pr.remove_words_outside_the_vocabulary(all_data, vocabulary)

    length_data = [[] for i in range(400)]
    length_logging = [[] for i in range(400)]
    length_path = [[] for i in range(400)]
    length_method = [[] for i in range(400)]

    for i, item in enumerate(standard_data):
        sentence_len = len(item)
        if sentence_len > 400:
            continue
        length_data[sentence_len-1].append(all_data[i])
        length_logging[sentence_len-1].append(all_logging[i])
        length_path[sentence_len-1].append(all_path[i])
        length_method[sentence_len-1].append(all_method[i])

    for i in range(400):
        zero, one = ev.return_the_number_of_label_data(length_logging[i])
        limit_num = min(zero, one)
        length_data[i], length_logging[i], length_path[i], length_method[i] = pr.balance_out_data(length_data[i], length_logging[i], length_path[i], length_method[i], limit_num)
        length_data_path = "input_data/end2/sentence_len/data1/"+input_file+"-code_sentence_len"+str(i+1)+"_frequency"+str(min_word_frequency_word2vec)+".json"
        pr.make_input_data_json(length_data_path, label_name, length_data[i], length_logging[i], length_path[i], length_method[i])

if input_file_num >= 2:
    pr.data_shuffle(all_data1, 1398)
    pr.data_shuffle(all_logging2, 1398)
    pr.data_shuffle(all_path2, 1398)
    pr.data_shuffle(all_method2, 1398)

    all_data1 = pr.remove_words_outside_the_vocabulary(all_data1, vocabulary)

    length_data = [[] for i in range(400)]
    length_logging = [[] for i in range(400)]
    length_path = [[] for i in range(400)]
    length_method = [[] for i in range(400)]

    for i, item in enumerate(standard_data):
        sentence_len = len(item)
        if sentence_len > 400:
            continue
        length_data[sentence_len-1].append(all_data1[i])
        length_logging[sentence_len-1].append(all_logging2[i])
        length_path[sentence_len-1].append(all_path2[i])
        length_method[sentence_len-1].append(all_method2[i])

    for i in range(400):
        zero, one = ev.return_the_number_of_label_data(length_logging[i])
        limit_num = min(zero, one)
        length_data[i], length_logging[i], length_path[i], length_method[i] = pr.balance_out_data(length_data[i], length_logging[i], length_path[i], length_method[i], limit_num)
        length_data_path = "input_data/end2/sentence_len/data1/"+input_file+"-AST_sentence_len"+str(i+1)+"_frequency"+str(min_word_frequency_word2vec)+".json"
        pr.make_input_data_json(length_data_path, label_name, length_data[i], length_logging[i], length_path[i], length_method[i])
    
    
if input_file_num >= 3:
    pr.data_shuffle(all_data3, 1398)
    pr.data_shuffle(all_logging3, 1398)
    pr.data_shuffle(all_path3, 1398)
    pr.data_shuffle(all_method3, 1398)

    all_data3 = pr.remove_words_outside_the_vocabulary(all_data3, vocabulary)

    length_data = [[] for i in range(400)]
    length_logging = [[] for i in range(400)]
    length_path = [[] for i in range(400)]
    length_method = [[] for i in range(400)]

    for i, item in enumerate(standard_data):
        sentence_len = len(item)
        if sentence_len > 400:
            continue
        length_data[sentence_len-1].append(all_data3[i])
        length_logging[sentence_len-1].append(all_logging3[i])
        length_path[sentence_len-1].append(all_path3[i])
        length_method[sentence_len-1].append(all_method3[i])

    for i in range(400):
        zero, one = ev.return_the_number_of_label_data(length_logging[i])
        limit_num = min(zero, one)
        length_data[i], length_logging[i], length_path[i], length_method[i] = pr.balance_out_data(length_data[i], length_logging[i], length_path[i], length_method[i], limit_num)
        length_data_path = "input_data/end2/sentence_len/data1/"+input_file+"-CAST_sentence_len"+str(i+1)+"_frequency"+str(min_word_frequency_word2vec)+".json"
        pr.make_input_data_json(length_data_path, label_name, length_data[i], length_logging[i], length_path[i], length_method[i])
    
if input_file_num >= 4:
    pr.data_shuffle(all_data4, 1398)
    pr.data_shuffle(all_logging4, 1398)
    pr.data_shuffle(all_path4, 1398)
    pr.data_shuffle(all_method4, 1398)

    all_data4 = pr.remove_words_outside_the_vocabulary(all_data4, vocabulary)

    length_data = [[] for i in range(400)]
    length_logging = [[] for i in range(400)]
    length_path = [[] for i in range(400)]
    length_method = [[] for i in range(400)]

    for i, item in enumerate(standard_data):
        sentence_len = len(item)
        if sentence_len > 400:
            continue
        length_data[sentence_len-1].append(all_data4[i])
        length_logging[sentence_len-1].append(all_logging4[i])
        length_path[sentence_len-1].append(all_path4[i])
        length_method[sentence_len-1].append(all_method4[i])

    for i in range(400):
        zero, one = ev.return_the_number_of_label_data(length_logging[i])
        limit_num = min(zero, one)
        length_data[i], length_logging[i], length_path[i], length_method[i] = pr.balance_out_data(length_data[i], length_logging[i], length_path[i], length_method[i], limit_num)
        length_data_path = "input_data/end2/sentence_len/data1/"+input_file+"-depth_num_sentence_len"+str(i+1)+"_frequency"+str(min_word_frequency_word2vec)+".json"
        pr.make_input_data_json(length_data_path, label_name, length_data[i], length_logging[i], length_path[i], length_method[i])

