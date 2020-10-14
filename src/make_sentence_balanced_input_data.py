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
min_word_frequency_word2vec = int(sys.argv[input_file_num+4])

#2. Classifier hyperparameters
st_min_sentence_len = int(sys.argv[input_file_num+6])
st_max_sentence_len = int(sys.argv[input_file_num+7])
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
        wordvec_model_name += input_filename[i][len(input_file)+1]
    wordvec_model_name += '_frequency'+str(min_word_frequency_word2vec)
standard_input_filename = input_file + '-' + standard_input_type
java_auto_logging_json = []
for i in range(input_file_num):
    java_auto_logging_json.append("../../MakeJSON/output/Logging/"+input_filename[i]+".json")
standard_java_auto_logging_json = '../../MakeJSON/output/Logging/'+standard_input_filename+'.json'
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

all_data = [0 for _ in range(input_file_num)]
all_logging = [0 for _ in range(input_file_num)]
all_path = [0 for _ in range(input_file_num)]
all_method = [0 for _ in range(input_file_num)]
for i in range(input_file_num):
    all_data[i], all_logging[i], all_path[i], all_method[i] = pr.open_data(java_auto_logging_json[i], label_name)

# Load the word2vec model and vocabulary
model_name = "../Wordvec_Model/"+wordvec_model_name+".model"
standard_model_name = "../Wordvec_Model/"+standard_wordvec_model_name+".model"
wordvec_model, vocabulary = pr.load_word2vec_model(model_name)
standard_wordvec_model, standard_vocabulary = pr.load_word2vec_model(standard_model_name)

#========================================================================================
# Split cross validation sets and perform deep learning + softmax based classification
#========================================================================================
seed = int(sys.argv[input_file_num+5])
pr.data_shuffle(standard_data, seed)

for i in range(input_file_num):
    pr.data_shuffle(all_data[i], seed)
    pr.data_shuffle(all_logging[i], seed)
    pr.data_shuffle(all_path[i], seed)
    pr.data_shuffle(all_method[i], seed)
   
    all_data[i] = pr.remove_words_outside_the_vocabulary(all_data[i], vocabulary)

    length_data = [[] for i in range(400)]
    length_logging = [[] for i in range(400)]
    length_path = [[] for i in range(400)]
    length_method = [[] for i in range(400)]

    for j, item in enumerate(standard_data):
        sentence_len = len(item)
        if sentence_len > 400:
            continue
        length_data[sentence_len-1].append(all_data[i][j])
        length_logging[sentence_len-1].append(all_logging[i][j])
        length_path[sentence_len-1].append(all_path[i][j])
        length_method[sentence_len-1].append(all_method[i][j])

    for i2 in range(400):
        zero, one = ev.return_the_number_of_label_data(length_logging[i2])
        limit_num = min(zero, one)
        length_data[i2], length_logging[i2], length_path[i2], length_method[i2] = pr.balance_out_data(length_data[i2], length_logging[i2], length_path[i2], length_method[i2], limit_num)
        length_data_path = "../input_data/sentence_len/data1/"+input_file+'/'+input_filename[i]+"_sentence_len"+str(i2+1)+"_frequency"+str(min_word_frequency_word2vec)+".json"
        pr.make_input_data_json(length_data_path, label_name, length_data[i2], length_logging[i2], length_path[i2], length_method[i2])

