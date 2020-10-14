#!/usr/bin/env python
# coding: utf-8

# ##### Copyright 2019 The TensorFlow Authors.

# In[1]:


#@title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# # Neural machine translation with attention

# <table class="tfo-notebook-buttons" align="left">
#   <td>
#     <a target="_blank" href="https://www.tensorflow.org/tutorials/text/nmt_with_attention">
#     <img src="https://www.tensorflow.org/images/tf_logo_32px.png" />
#     View on TensorFlow.org</a>
#   </td>
#   <td>
#     <a target="_blank" href="https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/text/nmt_with_attention.ipynb">
#     <img src="https://www.tensorflow.org/images/colab_logo_32px.png" />
#     Run in Google Colab</a>
#   </td>
#   <td>
#     <a target="_blank" href="https://github.com/tensorflow/docs/blob/master/site/en/tutorials/text/nmt_with_attention.ipynb">
#     <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
#     View source on GitHub</a>
#   </td>
#   <td>
#     <a href="https://storage.googleapis.com/tensorflow_docs/docs/site/en/tutorials/text/nmt_with_attention.ipynb"><img src="https://www.tensorflow.org/images/download_logo_32px.png" />Download notebook</a>
#   </td>
# </table>

# This notebook trains a sequence to sequence (seq2seq) model for Spanish to English translation. This is an advanced example that assumes some knowledge of sequence to sequence models.
# 
# After training the model in this notebook, you will be able to input a Spanish sentence, such as *"¿todavia estan en casa?"*, and return the English translation: *"are you still at home?"*
# 
# The translation quality is reasonable for a toy example, but the generated attention plot is perhaps more interesting. This shows which parts of the input sentence has the model's attention while translating:
# 
# <img src="https://tensorflow.org/images/spanish-english.png" alt="spanish-english attention plot">
# 
# Note: This example takes approximately 10 mintues to run on a single P100 GPU.

# In[2]:


from __future__ import absolute_import, division, print_function, unicode_literals

try:
  # %tensorflow_version only exists in Colab.
  get_ipython().run_line_magic('tensorflow_version', '2.x')
except Exception:
  pass
import tensorflow as tf

import preprocessing_data as pr
import evaluation as ev

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.model_selection import train_test_split

import unicodedata
import re
import numpy as np
import os
import io
import time
import sys
from copy import deepcopy
import nltk
from nltk.translate.bleu_score import sentence_bleu

tf.enable_eager_execution()
# ## Download and prepare the dataset
# 
# We'll use a language dataset provided by http://www.manythings.org/anki/. This dataset contains language translation pairs in the format:
# 
# ```
# May I borrow this book?	¿Puedo tomar prestado este libro?
# ```
# 
# There are a variety of languages available, but we'll use the English-Spanish dataset. For convenience, we've hosted a copy of this dataset on Google Cloud, but you can also download your own copy. After downloading the dataset, here are the steps we'll take to prepare the data:
# 
# 1. Add a *start* and *end* token to each sentence.
# 2. Clean the sentences by removing special characters.
# 3. Create a word index and reverse word index (dictionaries mapping from word → id and id → word).
# 4. Pad each sentence to a maximum length.

# In[3]:

# Input data define.
st_min_sentence_len = int(sys.argv[5])
st_max_sentence_len = int(sys.argv[6])
min_word_frequency_word2vec = int(sys.argv[4])
label_name = "isLogged"

# Input file name
input_file = sys.argv[1]
input_type = sys.argv[2]
target_type = sys.argv[3]
input_filename = input_file + '-' + input_type
target_filename = input_file + '-' + target_type

filename = "Attention-S2S-"+input_file + '_' + input_type + "-"+target_type+ "_sentence_balanced_min"+str(st_min_sentence_len)+"_max"+str(st_max_sentence_len)+"_frequency"+str(min_word_frequency_word2vec)

# Download the file
path_to_zip = tf.keras.utils.get_file(
    'spa-eng.zip', origin='http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip',
    extract=True)

path_to_file = os.path.dirname(path_to_zip)+"/spa-eng/spa.txt"


# In[4]:


# Converts the unicode file to ascii
def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn')


def preprocess_sentence(w):
    '''
    w = unicode_to_ascii(w.lower().strip())

    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ."
    # Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)

    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)

    w = w.rstrip().strip()
    '''
#w = unicode_to_ascii(w.lower().strip())
    w_tokens = nltk.word_tokenize(w)
    w = ' '.join(w_tokens)

    # adding a start and an end token to the sentence
    # so that the model know when to start and stop predicting.
    w = '<start> ' + w + ' <end>'
    return w


# In[5]:


# In[6]:


# 1. Remove the accents
# 2. Clean the sentences
# 3. Return word pairs in the format: [ENGLISH, SPANISH]
def create_dataset(path, num_examples):
#    lines = io.open(path, encoding='UTF-8').read().strip().split('\n')

#   word_pairs = [[preprocess_sentence(w) for w in l.split('\t')]  for l in lines[:num_examples]]

    # Load input data.
    for i in range(st_min_sentence_len, st_max_sentence_len+1):
        java_auto_logging_json = path+input_filename+"_sentence_len"+str(i)+"_frequency"+str(min_word_frequency_word2vec)+".json"
        temp_data, temp_logging, temp_path, temp_method = pr.open_data(java_auto_logging_json, label_name)
        if i == st_min_sentence_len:
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
    print("Input data")
    print("zero : ", zero)
    print("one : ", one)
    print()

    for i in range(st_min_sentence_len, st_max_sentence_len+1):
        java_auto_logging_json = path+target_filename+"_sentence_len"+str(i)+"_frequency"+str(min_word_frequency_word2vec)+".json"
        temp_data, temp_logging, temp_path, temp_method = pr.open_data(java_auto_logging_json, label_name)
        if i == st_min_sentence_len:
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
    print("Target data")
    print("zero : ", zero)
    print("one : ", one)
    print()

    word_pairs = []
    for i, item in enumerate(all_data):
        sentence = ''
        sentence2 = ''
        for data in item:
            sentence += data + ' '
        for data2 in all_data2[i]:
            sentence2 += data2 + ' '
        word_pairs.append([preprocess_sentence(sentence2), preprocess_sentence(sentence)])

    return zip(*word_pairs)


# In[7]:
# In[8]:


def max_length(tensor):
    return max(len(t) for t in tensor)


# In[9]:


def tokenize(lang):
  lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(
      filters='', lower=False)
  lang_tokenizer.fit_on_texts(lang)

  tensor = lang_tokenizer.texts_to_sequences(lang)

  tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor,
                                                         padding='post')

  return tensor, lang_tokenizer


# In[10]:


def load_dataset(path, num_examples=None):
    # creating cleaned input, output pairs
    targ_lang, inp_lang = create_dataset(path, num_examples)

    input_tensor, inp_lang_tokenizer = tokenize(inp_lang)
    target_tensor, targ_lang_tokenizer = tokenize(targ_lang)

    return input_tensor, target_tensor, inp_lang_tokenizer, targ_lang_tokenizer


# ### Limit the size of the dataset to experiment faster (optional)
# 
# Training on the complete dataset of >100,000 sentences will take a long time. To train faster, we can limit the size of the dataset to 30,000 sentences (of course, translation quality degrades with less data):

# In[11]:


# Try experimenting with the size of that dataset
num_examples = 30000
path_to_file = "../JavaAutoLogging/input_data/end2/sentence_len/data1/"
input_tensor, target_tensor, inp_lang, targ_lang = load_dataset(path_to_file, num_examples)

# Calculate max_length of the target tensors
max_length_targ, max_length_inp = max_length(target_tensor), max_length(input_tensor)
print(max_length_targ, max_length_inp)

# In[12]:


# Creating training and validation sets using an 80-20 split
input_tensor_train, input_tensor_test, target_tensor_train, target_tensor_test = train_test_split(input_tensor, target_tensor, test_size=0.2)

# Show length
print(len(input_tensor_train), len(target_tensor_train), len(input_tensor_test), len(target_tensor_test))

# In[13]:


def convert(lang, tensor):
  for t in tensor:
    if t!=0:
      print ("%d ----> %s" % (t, lang.index_word[t]))


# In[14]:

'''
print ("Input Language; index to word mapping")
convert(inp_lang, input_tensor_train[0])
print ()
print ("Target Language; index to word mapping")
convert(targ_lang, target_tensor_train[0])
'''
# ### Create a tf.data dataset

# In[15]:


BUFFER_SIZE = len(input_tensor_train)
BATCH_SIZE = 64
steps_per_epoch = len(input_tensor_train)//BATCH_SIZE
embedding_dim = 256
units = 1024
vocab_inp_size = len(inp_lang.word_index)+1
vocab_tar_size = len(targ_lang.word_index)+1

dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

print("Input vocabulary : ", len(inp_lang.word_index))
for i, data in enumerate(inp_lang.word_index):
    print(i, data)
print()
print("Target vocabulary : ", len(targ_lang.word_index))
for i, data in enumerate(targ_lang.word_index):
    print(i, data)
# In[16]:


example_input_batch, example_target_batch = next(iter(dataset))
example_input_batch.shape, example_target_batch.shape


# ## Write the encoder and decoder model
# 
# Implement an encoder-decoder model with attention which you can read about in the TensorFlow [Neural Machine Translation (seq2seq) tutorial](https://github.com/tensorflow/nmt). This example uses a more recent set of APIs. This notebook implements the [attention equations](https://github.com/tensorflow/nmt#background-on-the-attention-mechanism) from the seq2seq tutorial. The following diagram shows that each input words is assigned a weight by the attention mechanism which is then used by the decoder to predict the next word in the sentence. The below picture and formulas are an example of attention mechanism from [Luong's paper](https://arxiv.org/abs/1508.04025v5). 
# 
# <img src="https://www.tensorflow.org/images/seq2seq/attention_mechanism.jpg" width="500" alt="attention mechanism">
# 
# The input is put through an encoder model which gives us the encoder output of shape *(batch_size, max_length, hidden_size)* and the encoder hidden state of shape *(batch_size, hidden_size)*.
# 
# Here are the equations that are implemented:
# 
# <img src="https://www.tensorflow.org/images/seq2seq/attention_equation_0.jpg" alt="attention equation 0" width="800">
# <img src="https://www.tensorflow.org/images/seq2seq/attention_equation_1.jpg" alt="attention equation 1" width="800">
# 
# This tutorial uses [Bahdanau attention](https://arxiv.org/pdf/1409.0473.pdf) for the encoder. Let's decide on notation before writing the simplified form:
# 
# * FC = Fully connected (dense) layer
# * EO = Encoder output
# * H = hidden state
# * X = input to the decoder
# 
# And the pseudo-code:
# 
# * `score = FC(tanh(FC(EO) + FC(H)))`
# * `attention weights = softmax(score, axis = 1)`. Softmax by default is applied on the last axis but here we want to apply it on the *1st axis*, since the shape of score is *(batch_size, max_length, hidden_size)*. `Max_length` is the length of our input. Since we are trying to assign a weight to each input, softmax should be applied on that axis.
# * `context vector = sum(attention weights * EO, axis = 1)`. Same reason as above for choosing axis as 1.
# * `embedding output` = The input to the decoder X is passed through an embedding layer.
# * `merged vector = concat(embedding output, context vector)`
# * This merged vector is then given to the GRU
# 
# The shapes of all the vectors at each step have been specified in the comments in the code:

# In[17]:


class Encoder(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
    super(Encoder, self).__init__()
    self.batch_sz = batch_sz
    self.enc_units = enc_units
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(self.enc_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')

  def call(self, x, hidden):
    x = self.embedding(x)
    output, state = self.gru(x, initial_state = hidden)
    return output, state

  def initialize_hidden_state(self):
    return tf.zeros((self.batch_sz, self.enc_units))


# In[18]:


encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)

# sample input
sample_hidden = encoder.initialize_hidden_state()
sample_output, sample_hidden = encoder(example_input_batch, sample_hidden)
print ('Encoder output shape: (batch size, sequence length, units) {}'.format(sample_output.shape))
print ('Encoder Hidden state shape: (batch size, units) {}'.format(sample_hidden.shape))


# In[19]:


class BahdanauAttention(tf.keras.layers.Layer):
  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)

  def call(self, query, values):
    # hidden shape == (batch_size, hidden size)
    # hidden_with_time_axis shape == (batch_size, 1, hidden size)
    # we are doing this to perform addition to calculate the score
    hidden_with_time_axis = tf.expand_dims(query, 1)

    # score shape == (batch_size, max_length, 1)
    # we get 1 at the last axis because we are applying score to self.V
    # the shape of the tensor before applying self.V is (batch_size, max_length, units)
    score = self.V(tf.nn.tanh(
        self.W1(values) + self.W2(hidden_with_time_axis)))

    # attention_weights shape == (batch_size, max_length, 1)
    attention_weights = tf.nn.softmax(score, axis=1)

    # context_vector shape after sum == (batch_size, hidden_size)
    context_vector = attention_weights * values
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights


# In[20]:


attention_layer = BahdanauAttention(10)
attention_result, attention_weights = attention_layer(sample_hidden, sample_output)

print("Attention result shape: (batch size, units) {}".format(attention_result.shape))
print("Attention weights shape: (batch_size, sequence_length, 1) {}".format(attention_weights.shape))


# In[21]:


class Decoder(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
    super(Decoder, self).__init__()
    self.batch_sz = batch_sz
    self.dec_units = dec_units
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(self.dec_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')
    self.fc = tf.keras.layers.Dense(vocab_size)

    # used for attention
    self.attention = BahdanauAttention(self.dec_units)

  def call(self, x, hidden, enc_output):
    # enc_output shape == (batch_size, max_length, hidden_size)
    context_vector, attention_weights = self.attention(hidden, enc_output)

    # x shape after passing through embedding == (batch_size, 1, embedding_dim)
    x = self.embedding(x)

    # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
    x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

    # passing the concatenated vector to the GRU
    output, state = self.gru(x)

    # output shape == (batch_size * 1, hidden_size)
    output = tf.reshape(output, (-1, output.shape[2]))

    # output shape == (batch_size, vocab)
    x = self.fc(output)

    return x, state, attention_weights


# In[22]:


decoder = Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE)

sample_decoder_output, _, _ = decoder(tf.random.uniform((64, 1)),
                                      sample_hidden, sample_output)

print ('Decoder output shape: (batch_size, vocab size) {}'.format(sample_decoder_output.shape))


# ## Define the optimizer and the loss function

# In[23]:


optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

def loss_function(real, pred):
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = loss_object(real, pred)

  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask

  return tf.reduce_mean(loss_)


# ## Checkpoints (Object-based saving)

# In[24]:


checkpoint_dir = './training_checkpoints/'+filename
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=encoder,
                                 decoder=decoder)

'''
# ## Training
# 
# 1. Pass the *input* through the *encoder* which return *encoder output* and the *encoder hidden state*.
# 2. The encoder output, encoder hidden state and the decoder input (which is the *start token*) is passed to the decoder.
# 3. The decoder returns the *predictions* and the *decoder hidden state*.
# 4. The decoder hidden state is then passed back into the model and the predictions are used to calculate the loss.
# 5. Use *teacher forcing* to decide the next input to the decoder.
# 6. *Teacher forcing* is the technique where the *target word* is passed as the *next input* to the decoder.
# 7. The final step is to calculate the gradients and apply it to the optimizer and backpropagate.

# In[25]:


@tf.function
def train_step(inp, targ, enc_hidden):
  loss = 0

  with tf.GradientTape() as tape:
    enc_output, enc_hidden = encoder(inp, enc_hidden)

    dec_hidden = enc_hidden

    dec_input = tf.expand_dims([targ_lang.word_index['<start>']] * BATCH_SIZE, 1)

    # Teacher forcing - feeding the target as the next input
    for t in range(1, targ.shape[1]):
      # passing enc_output to the decoder
      predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)

      loss += loss_function(targ[:, t], predictions)

      # using teacher forcing
      dec_input = tf.expand_dims(targ[:, t], 1)

  batch_loss = (loss / int(targ.shape[1]))

  variables = encoder.trainable_variables + decoder.trainable_variables

  gradients = tape.gradient(loss, variables)

  optimizer.apply_gradients(zip(gradients, variables))

  return batch_loss


# In[26]:


EPOCHS = 15

for epoch in range(EPOCHS):
  start = time.time()

  enc_hidden = encoder.initialize_hidden_state()
  total_loss = 0

  for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
    batch_loss = train_step(inp, targ, enc_hidden)
    total_loss += batch_loss

    if batch % 100 == 0:
        print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                     batch,
                                                     batch_loss.numpy()))
  # saving (checkpoint) the model every 2 epochs and last epochs
  if (epoch + 1) % 2 == 0 or epoch == EPOCHS-1:
    checkpoint.save(file_prefix = checkpoint_prefix)

  print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                      total_loss / steps_per_epoch))
  print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

'''

# ## Translate
# 
# * The evaluate function is similar to the training loop, except we don't use *teacher forcing* here. The input to the decoder at each time step is its previous predictions along with the hidden state and the encoder output.
# * Stop predicting when the model predicts the *end token*.
# * And store the *attention weights for every time step*.
# 
# Note: The encoder output is calculated only once for one input.

# In[27]:


def evaluate(sentence):
    attention_plot = np.zeros((max_length_targ, max_length_inp))

    sentence = preprocess_sentence(sentence)

#inputs = [inp_lang.word_index[i] for i in sentence.split(' ')]
    inputs = []
    for i in sentence.split(' '):
        if i in inp_lang.word_index:
            inputs.append(inp_lang.word_index[i])

    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
                                                           maxlen=max_length_inp,
                                                           padding='post')
    inputs = tf.convert_to_tensor(inputs)

    result = ''

    hidden = [tf.zeros((1, units))]
    enc_out, enc_hidden = encoder(inputs, hidden)

    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([targ_lang.word_index['<start>']], 0)

    for t in range(max_length_targ):
        predictions, dec_hidden, attention_weights = decoder(dec_input,
                                                             dec_hidden,
                                                             enc_out)

        # storing the attention weights to plot later on
        attention_weights = tf.reshape(attention_weights, (-1, ))
        attention_plot[t] = attention_weights.numpy()

        predicted_id = tf.argmax(predictions[0]).numpy()

        result += targ_lang.index_word[predicted_id] + ' '

        if targ_lang.index_word[predicted_id] == '<end>':
            return result, sentence, attention_plot

        # the predicted ID is fed back into the model
        dec_input = tf.expand_dims([predicted_id], 0)

    return result, sentence, attention_plot


# In[28]:


# function for plotting the attention weights
def plot_attention(attention, sentence, predicted_sentence, count):
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(attention, cmap='viridis')

    fontdict = {'fontsize': 14}

    ax.set_xticklabels([''] + sentence, fontdict=fontdict, rotation=90)
    ax.set_yticklabels([''] + predicted_sentence, fontdict=fontdict)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.savefig("plot"+str(count))


# In[29]:

#count = 0
def translate(sentence):
    result, sentence, attention_plot = evaluate(sentence)

    f = open("Load/load-predict-"+filename+".txt", 'a')

    result = result.strip()
    result = "<start> " + result
    f.write(result + '\r\n')

    f.close()
#    print('Input: %s' % (sentence))
#    print('Predicted translation: {}'.format(result))

#    count += 1
    attention_plot = attention_plot[:len(result.split(' ')), :len(sentence.split(' '))]
#plot_attention(attention_plot, sentence.split(' '), result.split(' '), count)

    return result

# ## Restore the latest checkpoint and test

# In[30]:


# restoring the latest checkpoint in checkpoint_dir
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

# test translation. 
#'''
correct = 0
first = True
score = []
sentence_arr = []
sentence2_arr = []
f_ans = open("Load/load-answer-"+filename+".txt", 'w')
for i, item in enumerate(input_tensor_test):
    sentence = ''
    sentence2 = ''
    for j, data in enumerate(item):
        if data == 0:
            break
        sentence += inp_lang.index_word[data] + ' '
    for j2, data2 in enumerate(target_tensor_test[i]):
        if data2 == 0:
            break
        sentence2 += targ_lang.index_word[data2] + ' '
    sentence2 = sentence2.strip()
    if first:
        f = open("Load/load-predict-"+filename+".txt", 'w')
        f.close()
        first = False

    sentence = translate(sentence)
    f_ans.write(sentence2 + '\r\n')
    if sentence == sentence2:
        correct += 1
    sentence_arr.append(sentence)
    sentence2_arr.append(sentence2)
    score.append(sentence_bleu(sentence, sentence2))
print("BLEU : ",score)
print("Avg of BLEU : ", sum(score, 0)/len(score))
print("BLEU all : ", sentence_bleu(sentence_arr, sentence2_arr))

print("Accuracy : ", (correct/len(input_tensor_test))*100)

f_ans.close()
sys.exit(1)
#'''
# Load test data.
# Part of train data.
print("Part of train data.")
min = 6
max = 10
correct = 0
first = True
score = []
sentence_arr = []
sentence2_arr = []
f_ans = open("Load/load-answer-"+filename+".txt", 'w')
for i in range(min, max):
    java_auto_logging_json = "../JavaAutoLogging/input_data/end2/sentence_len/data1/"+input_filename+"_sentence_len"+str(i)+"_frequency"+str(min_word_frequency_word2vec)+".json"
    temp_data, temp_logging, temp_path, temp_method = pr.open_data(java_auto_logging_json, label_name)
    if i == min:
        test_data = deepcopy(temp_data)
    else:
        test_data += temp_data

    java_auto_logging_json2 = "../JavaAutoLogging/input_data/end2/sentence_len/data1/"+target_filename+"_sentence_len"+str(i)+"_frequency"+str(min_word_frequency_word2vec)+".json"
    temp_data2, temp_logging2, temp_path2, temp_method2 = pr.open_data(java_auto_logging_json2, label_name)
    if i == min:
        ans_data = deepcopy(temp_data2)
    else:
        ans_data += temp_data2

for i, item in enumerate(test_data):
    sentence = ''
    sentence2 = ''
    for data in item:
        sentence += data + ' '
    for data2 in ans_data[i]:
        sentence2 += data2 + ' '
    sentence2 = preprocess_sentence(sentence2)
    f_ans.write(sentence2 + '\r\n')

    if first:
        f = open("Load/load-predict-"+filename+".txt", 'w')
        f.close()
        first = False
    sentence = translate(sentence)
    if sentence == sentence2:
        correct += 1
    sentence_arr.append(sentence)
    sentence2_arr.append(sentence2)
    score.append(sentence_bleu(sentence, sentence2))
print("Accuracy : ", (correct/len(test_data))*100)
print("BLEU : ",score)
print("Avg of BLEU : ", sum(score, 0)/len(score))
print("BLEU all : ", sentence_bleu(sentence_arr, sentence2_arr))

f_ans.close()
sys.exit(1)

'''
# Out of train data.
print("Out of train data.")
min = 6
max = 10
correct = 0
first = True
score = []
sentence_arr = []
sentence2_arr = []
f_ans = open("Load/load-answer-"+filename+".txt", 'w')
for i in range(20, 30):
    java_auto_logging_json = "../JavaAutoLogging/input_data/end2/sentence_len/data1/"+input_filename+"_sentence_len"+str(i)+"_frequency"+str(min_word_frequency_word2vec)+".json"
    temp_data, temp_logging, temp_path, temp_method = pr.open_data(java_auto_logging_json, label_name)
    if i == 20:
        test_data = deepcopy(temp_data)
    else:
        test_data += temp_data

    java_auto_logging_json2 = "../JavaAutoLogging/input_data/end2/sentence_len/data1/"+target_filename+"_sentence_len"+str(i)+"_frequency"+str(min_word_frequency_word2vec)+".json"
    temp_data2, temp_logging2, temp_path2, temp_method2 = pr.open_data(java_auto_logging_json2, label_name)
    if i == min:
        ans_data = deepcopy(temp_data2)
    else:
        ans_data += temp_data2


    for i, item in enumerate(test_data):
        sentence = ''
        for data in item:
            sentence += data + ' '
        translate(sentence)

for i, item in enumerate(test_data):
    sentence = ''
    sentence2 = ''
    for data in item:
        sentence += data + ' '
    for data2 in ans_data[i]:
        sentence2 += data2 + ' '
    sentence2 = preprocess_sentence(sentence2)
    f_ans.write(sentence2 + '\r\n')

    if first:
        f = open("Load/load-predict-"+filename+".txt", 'w')
        f.close()
        first = False
    sentence = translate(sentence)
    print("BLEU test")
    print(sentence)
    print(sentence2)
    if sentence == sentence2:
        correct += 1
    sentence_arr.append(sentence)
    sentence2_arr.append(sentence2)
    score.append(sentence_bleu(sentence, sentence2))
print("Accuracy : ", (correct/len(test_data))*100)
print("BLEU : ",score)
print("Avg of BLEU : ", sum(score, 0)/len(score))
print("BLEU all : ", sentence_bleu(sentence_arr, sentence2_arr))
print("BLEU2 : ", sentence_bleu(list(map(lambda sen: sen.split(), sentence_arr)), list(map(lambda sen2: sen2.split(), sentence2_arr))))
'''
# In[31]:


#translate(u'hace mucho frio aqui.')


# In[32]:


#translate(u'esta es mi vida.')


# In[33]:


#translate(u'¿todavia estan en casa?')


# In[34]:


# wrong translation
#translate(u'trata de averiguarlo.')


# ## Next steps
# 
# * [Download a different dataset](http://www.manythings.org/anki/) to experiment with translations, for example, English to German, or English to French.
# * Experiment with training on a larger dataset, or using more epochs
# 
