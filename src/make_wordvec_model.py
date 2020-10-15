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
label = sys.argv[input_file_num+4]
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
    wordvec_model_name = input_filename[0] + "_frequency"+str(min_word_frequency_word2vec)
else:
    merge = ""
    for i in range(input_file_num):
        merge += input_filename[i][len(input_file)+1]
    wordvec_model_name = input_file + "-CAST_m_" + merge + '_frequency'+str(min_word_frequency_word2vec)

json_file_path = []
for i in range(input_file_num):
    json_file_path.append("../MakeJSON/output/%s/%s.json"%(label, input_filename[i]))

#========================================================================================
# Preprocess the java auto logging, extract the vocabulary and learn the word2vec representation
#========================================================================================

merge_data = []
label_data = [0 for _ in range(input_file_num)]
for i in range(input_file_num):
    temp, label_data[i], _, _ = pr.open_data(json_file_path[i], label_name)
    zero, one = ev.return_the_number_of_label_data(label_data[i])
    print("%sth data"%(i+1))
    print("zero : ", zero)
    print("one : ", one)
    print()
    merge_data += temp

# Learn the word2vec model and extract vocabulary
model_name = "../%s/Wordvec_Model/%s.model"%(label, wordvec_model_name)
wordvec_model, vocabulary = pr.train_word2vec_model(merge_data, min_word_frequency_word2vec, embed_size_word2vec, context_window_word2vec, model_name)

