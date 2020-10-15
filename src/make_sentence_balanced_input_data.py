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
label = sys.argv[input_file_num+8]
if label == "Logging":
    label_name = "isLogged"
elif label == "Exception":
    label_name = "isException"

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
json_file_path = []
for i in range(input_file_num):
    json_file_path.append("../MakeJSON/output/%s/%s.json"%(label, input_filename[i]))
standard_json_file_path = '../MakeJSON/output/%s/%s.json'%(label, standard_input_filename)
standard_wordvec_model_name = standard_input_filename + '_frequency'+str(min_word_frequency_word2vec)

#========================================================================================
# Preprocess the java auto logging, extract the vocabulary and learn the word2vec representation
#========================================================================================

standard_data, standard_label, standard_path, standard_method = pr.open_data(standard_json_file_path, label_name)

zero, one = ev.return_the_number_of_label_data(standard_label)
print("Standard data")
print("zero : ", zero)
print("one : ", one)
print()

all_data = [0 for _ in range(input_file_num)]
all_label = [0 for _ in range(input_file_num)]
all_path = [0 for _ in range(input_file_num)]
all_method = [0 for _ in range(input_file_num)]
for i in range(input_file_num):
    all_data[i], all_label[i], all_path[i], all_method[i] = pr.open_data(json_file_path[i], label_name)

# Load the word2vec model and vocabulary
model_name = "../%s/Wordvec_Model/%s.model"%(label, wordvec_model_name)
standard_model_name = "../%s/Wordvec_Model/%s.model"%(label, standard_wordvec_model_name)
wordvec_model, vocabulary = pr.load_word2vec_model(model_name)
standard_wordvec_model, standard_vocabulary = pr.load_word2vec_model(standard_model_name)


#========================================================================================
# Split cross validation sets and perform deep learning + softmax based classification
#========================================================================================
seed = int(sys.argv[input_file_num+5])
pr.data_shuffle(standard_data, seed)

standard_data = pr.remove_words_outside_the_vocabulary(standard_data, standard_vocabulary)
for i in range(input_file_num):
    pr.data_shuffle(all_data[i], seed)
    pr.data_shuffle(all_label[i], seed)
    pr.data_shuffle(all_path[i], seed)
    pr.data_shuffle(all_method[i], seed)
   
    all_data[i] = pr.remove_words_outside_the_vocabulary(all_data[i], vocabulary)

    length_data = [[] for i in range(400)]
    length_label = [[] for i in range(400)]
    length_path = [[] for i in range(400)]
    length_method = [[] for i in range(400)]

    for j, item in enumerate(standard_data):
        sentence_len = len(item)
        if sentence_len > 400:
            continue
        length_data[sentence_len-1].append(all_data[i][j])
        length_label[sentence_len-1].append(all_label[i][j])
        length_path[sentence_len-1].append(all_path[i][j])
        length_method[sentence_len-1].append(all_method[i][j])

    for i2 in range(400):
        zero, one = ev.return_the_number_of_label_data(length_label[i2])
        limit_num = min(zero, one)
        length_data[i2], length_label[i2], length_path[i2], length_method[i2] = pr.balance_out_data(length_data[i2], length_label[i2], length_path[i2], length_method[i2], limit_num)
        length_data_path = "../%s/Data/sentence_len/data2/%s/%s_sentence_len%s_frequency%s.json"%(label, input_file, input_filename[i], str(i2+1), str(min_word_frequency_word2vec))
        pr.make_input_data_json(length_data_path, label_name, length_data[i2], length_label[i2], length_path[i2], length_method[i2])

