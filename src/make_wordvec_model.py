# Required dependencies
# 1. NLTK
# 2. Gensim for word2vec

import preprocessing_data as pr
import model as md
import evaluation as ev

import sys
from copy import deepcopy

input_file_num = int(sys.argv[2])
#========================================================================================
# Initializing Hyper parameter
#========================================================================================
#1. Word2vec parameters
min_word_frequency_word2vec = int(sys.argv[input_file_num+3])
embed_size_word2vec = 200
context_window_word2vec = 5

#2. Classifier hyperparameters
label_name = "isLogged"

#3. Input file name
input_file = sys.argv[1]
input_filename = []
for i in range(input_file_num):
    input_filename.append(sys.argv[i+3])
    input_filename[i] = input_file + '-' + input_filename[i]
if input_file_num == 1:
    wordvec_model_name = input_filename[0] + "_frequency"+str(min_word_frequency_word2vec)
else:
    merge = ""
    for i in range(input_file_num):
        merge += input_filename[i][len(input_file)+1]
    wordvec_model_name = input_file + "-CAST_m_" + merge + '_frequency'+str(min_word_frequency_word2vec)

java_auto_logging_json = []
for i in range(input_file_num):
    java_auto_logging_json.append("../../MakeJSON/output/Logging/"+input_filename[i]+".json")

#========================================================================================
# Preprocess the java auto logging, extract the vocabulary and learn the word2vec representation
#========================================================================================

if input_file_num >= 1:
    merge_data, all_logging, all_path, all_method = pr.open_data(java_auto_logging_json[0], label_name)

    zero, one = ev.return_the_number_of_label_data(all_logging)
    print("First data")
    print("zero : ", zero)
    print("one : ", one)
    print()

if input_file_num >= 2:
    all_data2, all_logging2, all_path2, all_method2 = pr.open_data(java_auto_logging_json[1], label_name)

    zero, one = ev.return_the_number_of_label_data(all_logging2)
    print("Second data")
    print("zero : ", zero)
    print("one : ", one)
    print()
    merge_data += all_data2

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

'''
for i, item in enumerate(all_data):
    if all_logging[i] != all_logging3[i]:
#    if all_path[i] != all_path3[i]:# or all_path2[i] != all_path3[i]:
#       if all_method[i] != all_method3[i]:# or all_method2[i] != all_method3[i]:
        print(all_data[i], all_path[i], all_method[i])
        print(all_data3[i], all_path3[i], all_method3[i])
sys.exit(1)
'''
# Learn the word2vec model and extract vocabulary
model_name = "../Wordvec_Model/"+wordvec_model_name+".model"
wordvec_model, vocabulary = pr.train_word2vec_model(merge_data, min_word_frequency_word2vec, embed_size_word2vec, context_window_word2vec, model_name)

