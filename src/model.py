import sys
import json

import keras
import numpy as np
from keras.preprocessing import sequence
from keras.models import Model, model_from_json
from keras.layers import Dense, Dropout, Embedding, LSTM, Input, merge, Masking
from keras.optimizers import RMSprop, Adam

def load_model(model_json, model_h5):
    json_file = open(model_json, "r")
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights(model_h5)
    rms = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08)
    model.compile(loss='categorical_crossentropy', optimizer=rms, metrics=['accuracy'])
    return model

def stacking_merge_2_bidirectional_RNN(max_sentence_len, max_sentence_len2, embed_size_word2vec, LSTM_output_size):
    input1 = Input(shape=(max_sentence_len, embed_size_word2vec), dtype='float32')
    sequence = Masking(mask_value = 0.0)(input1)
    forwards_1 = LSTM(LSTM_output_size, name='forwards_1')(sequence)
    after_dp_forward_1 = Dropout(0.20, name='after_dp_forward_1')(forwards_1) 
    backwards_1 = LSTM(LSTM_output_size, go_backwards=True, name='backwords_1')(sequence)
    after_dp_backward_1 = Dropout(0.20, name='after_dp_backward_1')(backwards_1)         
    merged_1 = keras.layers.concatenate([after_dp_forward_1, after_dp_backward_1], axis=-1)
    after_dp_1= Dropout(0.5, name='after_dp_1')(merged_1)
                
    input2 = Input(shape=(max_sentence_len2, embed_size_word2vec), dtype='float32')
    sequence2 = Masking(mask_value = 0.0)(input2)
    forwards_2 = LSTM(LSTM_output_size, name='forwards_2')(sequence2)
    after_dp_forward_2 = Dropout(0.20, name='after_dp_forward_2')(forwards_2)
    backwards_2 = LSTM(LSTM_output_size, go_backwards=True, name='backwards_2')(sequence2)
    after_dp_backward_2 = Dropout(0.20, name='after_dp_backward_2')(backwards_2)
    merged_2 = keras.layers.concatenate([after_dp_forward_2, after_dp_backward_2], axis=-1)
    after_dp_2 = Dropout(0.5, name='after_dp_2')(merged_2)

    last_merge = keras.layers.concatenate([after_dp_1, after_dp_2], axis=-1)
    layer0 = Dense(256, activation='relu')(last_merge)
    layer1 = Dense(128, activation='relu')(layer0)
    layer2 = Dense(64, activation='relu')(layer1)
    layer3 = Dense(128, activation='relu')(layer2)
    layer4 = Dense(64, activation='relu')(layer3)
    output = Dense(2, activation='softmax', name='output')(layer4)

    model = Model(input=[input1, input2], output=output)            
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    return model

def stacking_merge_3_bidirectional_RNN(max_sentence_len, max_sentence_len2, max_sentence_len3, embed_size_word2vec, LSTM_output_size):
    input1 = Input(shape=(max_sentence_len, embed_size_word2vec), dtype='float32')
    sequence = Masking(mask_value = 0.0)(input1)
    forwards_1 = LSTM(LSTM_output_size, name='forwards_1')(sequence)
    after_dp_forward_1 = Dropout(0.20, name='after_dp_forward_1')(forwards_1) 
    backwards_1 = LSTM(LSTM_output_size, go_backwards=True, name='backwords_1')(sequence)
    after_dp_backward_1 = Dropout(0.20, name='after_dp_backward_1')(backwards_1)         
    merged_1 = keras.layers.concatenate([after_dp_forward_1, after_dp_backward_1], axis=-1)
    after_dp_1= Dropout(0.5, name='after_dp_1')(merged_1)
                
    input2 = Input(shape=(max_sentence_len2, embed_size_word2vec), dtype='float32')
    sequence2 = Masking(mask_value = 0.0)(input2)
    forwards_2 = LSTM(LSTM_output_size, name='forwards_2')(sequence2)
    after_dp_forward_2 = Dropout(0.20, name='after_dp_forward_2')(forwards_2)
    backwards_2 = LSTM(LSTM_output_size, go_backwards=True, name='backwards_2')(sequence2)
    after_dp_backward_2 = Dropout(0.20, name='after_dp_backward_2')(backwards_2)
    merged_2 = keras.layers.concatenate([after_dp_forward_2, after_dp_backward_2], axis=-1)
    after_dp_2 = Dropout(0.5, name='after_dp_2')(merged_2)

    input3 = Input(shape=(max_sentence_len3, embed_size_word2vec), dtype='float32')
    sequence3 = Masking(mask_value = 0.0)(input3)
    forwards_3 = LSTM(LSTM_output_size, name='forwards_3')(sequence3)
    after_dp_forward_3 = Dropout(0.20, name='after_dp_forward_3')(forwards_3)
    backwards_3 = LSTM(LSTM_output_size, go_backwards=True, name='backwards_3')(sequence3)
    after_dp_backward_3 = Dropout(0.20, name='after_dp_backward_3')(backwards_3)
    merged_3 = keras.layers.concatenate([after_dp_forward_3, after_dp_backward_3], axis=-1)
    after_dp_3 = Dropout(0.5, name='after_dp_3')(merged_3)

    merge1 = keras.layers.concatenate([after_dp_1, after_dp_2], axis=-1)
    last_merge = keras.layers.concatenate([merge1, after_dp_3], axis=-1)

    layer0 = Dense(384, activation='relu')(last_merge)
    layer1 = Dense(512, activation='relu')(layer0)
    layer2 = Dense(256, activation='relu')(layer1)
    layer3 = Dense(128, activation='relu')(layer2)
    layer4 = Dense(64, activation='relu')(layer3)
    layer5 = Dense(128, activation='relu')(layer4)
    layer6 = Dense(64, activation='relu')(layer5)
    output = Dense(2, activation='softmax', name='output')(layer6)

    model = Model(input=[input1, input2, input3], output=output) 
    rms = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08)
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    return model
def stacking_merge_4_bidirectional_RNN(max_sentence_len,max_sentence_len2, max_sentence_len3, max_sentence_len4, embed_size_word2vec, LSTM_output_size):
    input1 = Input(shape=(max_sentence_len, embed_size_word2vec), dtype='float32')
    sequence = Masking(mask_value = 0.0)(input1)
    forwards_1 = LSTM(LSTM_output_size, name='forwards_1')(sequence)
    after_dp_forward_1 = Dropout(0.20, name='after_dp_forward_1')(forwards_1) 
    backwards_1 = LSTM(LSTM_output_size, go_backwards=True, name='backwords_1')(sequence)
    after_dp_backward_1 = Dropout(0.20, name='after_dp_backward_1')(backwards_1)         
    merged_1 = keras.layers.concatenate([after_dp_forward_1, after_dp_backward_1], axis=-1)
    after_dp_1= Dropout(0.5, name='after_dp_1')(merged_1)
                
    input2 = Input(shape=(max_sentence_len2, embed_size_word2vec), dtype='float32')
    sequence2 = Masking(mask_value = 0.0)(input2)
    forwards_2 = LSTM(LSTM_output_size, name='forwards_2')(sequence2)
    after_dp_forward_2 = Dropout(0.20, name='after_dp_forward_2')(forwards_2)
    backwards_2 = LSTM(LSTM_output_size, go_backwards=True, name='backwards_2')(sequence2)
    after_dp_backward_2 = Dropout(0.20, name='after_dp_backward_2')(backwards_2)
    merged_2 = keras.layers.concatenate([after_dp_forward_2, after_dp_backward_2], axis=-1)
    after_dp_2 = Dropout(0.5, name='after_dp_2')(merged_2)

    input3 = Input(shape=(max_sentence_len3, embed_size_word2vec), dtype='float32')
    sequence3 = Masking(mask_value = 0.0)(input3)
    forwards_3 = LSTM(LSTM_output_size, name='forwards_3')(sequence3)
    after_dp_forward_3 = Dropout(0.20, name='after_dp_forward_3')(forwards_3)
    backwards_3 = LSTM(LSTM_output_size, go_backwards=True, name='backwards_3')(sequence3)
    after_dp_backward_3 = Dropout(0.20, name='after_dp_backward_3')(backwards_3)
    merged_3 = keras.layers.concatenate([after_dp_forward_3, after_dp_backward_3], axis=-1)
    after_dp_3 = Dropout(0.5, name='after_dp_3')(merged_3)

    input4 = Input(shape=(max_sentence_len4, embed_size_word2vec), dtype='float32')
    sequence4 = Masking(mask_value = 0.0)(input4)
    forwards_4 = LSTM(LSTM_output_size, name='forwards_4')(sequence4)
    after_dp_forward_4 = Dropout(0.20, name='after_dp_forward_4')(forwards_4)
    backwards_4 = LSTM(LSTM_output_size, go_backwards=True, name='backwards_4')(sequence4)
    after_dp_backward_4 = Dropout(0.20, name='after_dp_backward_4')(backwards_4)
    merged_4 = keras.layers.concatenate([after_dp_forward_4, after_dp_backward_4], axis=-1)
    after_dp_4 = Dropout(0.5, name='after_dp_4')(merged_4)

    merge1 = keras.layers.concatenate([after_dp_1, after_dp_2], axis=-1)
    merge2 = keras.layers.concatenate([merge1, after_dp_3], axis=-1)
    last_merge = keras.layers.concatenate([merge2, after_dp_4], axis=-1)

    layer0 = Dense(512, activation='relu')(last_merge)
    layer1 = Dense(768, activation='relu')(layer0)
    layer2 = Dense(512, activation='relu')(layer1)
    layer3 = Dense(256, activation='relu')(layer2)
    layer4 = Dense(128, activation='relu')(layer3)
    layer5 = Dense(256, activation='relu')(layer4)
    layer6 = Dense(128, activation='relu')(layer5)
    output = Dense(2, activation='softmax', name='output')(layer6)

    model = Model(input=[input1, input2, input3, input4], output=output) 
    rms = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08)
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    return model


# Construct the deep learning model
def bidirectional_RNN(max_sentence_len, embed_size_word2vec, LSTM_output_size):
    inputs = Input(shape=(max_sentence_len, embed_size_word2vec), dtype='float32')
    sequence = Masking(mask_value = 0.0)(inputs)
    forwards_1 = LSTM(LSTM_output_size)(sequence)
    after_dp_forward_4 = Dropout(0.20)(forwards_1) 
    backwards_1 = LSTM(LSTM_output_size, go_backwards=True)(sequence)
    after_dp_backward_4 = Dropout(0.20)(backwards_1)         
    merged = keras.layers.concatenate([after_dp_forward_4, after_dp_backward_4], axis=-1)
    after_dp = Dropout(0.5)(merged)
    output = Dense(2, activation='softmax')(after_dp)                
    model = Model(inputs, output)            
    rms = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08)
    model.compile(loss='categorical_crossentropy', optimizer=rms, metrics=['accuracy'])    

    return model

def merge_2_bidirectional_RNN(max_sentence_len, max_sentence_len2, embed_size_word2vec, LSTM_output_size):
    input1 = Input(shape=(max_sentence_len, embed_size_word2vec), dtype='float32')
    sequence = Masking(mask_value = 0.0)(input1)
    forwards_1 = LSTM(LSTM_output_size, name='forwards_1')(sequence)
    after_dp_forward_1 = Dropout(0.20, name='after_dp_forward_1')(forwards_1) 
    backwards_1 = LSTM(LSTM_output_size, go_backwards=True, name='backwords_1')(sequence)
    after_dp_backward_1 = Dropout(0.20, name='after_dp_backward_1')(backwards_1)         
    merged_1 = keras.layers.concatenate([after_dp_forward_1, after_dp_backward_1], axis=-1)
    after_dp_1= Dropout(0.5, name='after_dp_1')(merged_1)
                
    input2 = Input(shape=(max_sentence_len2, embed_size_word2vec), dtype='float32')
    sequence2 = Masking(mask_value = 0.0)(input2)
    forwards_2 = LSTM(LSTM_output_size, name='forwards_2')(sequence2)
    after_dp_forward_2 = Dropout(0.20, name='after_dp_forward_2')(forwards_2)
    backwards_2 = LSTM(LSTM_output_size, go_backwards=True, name='backwards_2')(sequence2)
    after_dp_backward_2 = Dropout(0.20, name='after_dp_backward_2')(backwards_2)
    merged_2 = keras.layers.concatenate([after_dp_forward_2, after_dp_backward_2], axis=-1)
    after_dp_2 = Dropout(0.5, name='after_dp_2')(merged_2)

    last_merge = keras.layers.concatenate([after_dp_1, after_dp_2], axis=-1)
    layer0 = Dense(256, activation='relu')(last_merge)
    layer1 = Dense(128, activation='relu')(layer0)
    layer2 = Dense(64, activation='relu')(layer1)
    output = Dense(2, activation='softmax', name='output')(layer2)

    model = Model(input=[input1, input2], output=output)            
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    return model

def merge_3_bidirectional_RNN(max_sentence_len, max_sentence_len2, max_sentence_len3, embed_size_word2vec, LSTM_output_size):
    input1 = Input(shape=(max_sentence_len, embed_size_word2vec), dtype='float32')
    sequence = Masking(mask_value = 0.0)(input1)
    forwards_1 = LSTM(LSTM_output_size, name='forwards_1')(sequence)
    after_dp_forward_1 = Dropout(0.20, name='after_dp_forward_1')(forwards_1) 
    backwards_1 = LSTM(LSTM_output_size, go_backwards=True, name='backwords_1')(sequence)
    after_dp_backward_1 = Dropout(0.20, name='after_dp_backward_1')(backwards_1)         
    merged_1 = keras.layers.concatenate([after_dp_forward_1, after_dp_backward_1], axis=-1)
    after_dp_1= Dropout(0.5, name='after_dp_1')(merged_1)
                
    input2 = Input(shape=(max_sentence_len2, embed_size_word2vec), dtype='float32')
    sequence2 = Masking(mask_value = 0.0)(input2)
    forwards_2 = LSTM(LSTM_output_size, name='forwards_2')(sequence2)
    after_dp_forward_2 = Dropout(0.20, name='after_dp_forward_2')(forwards_2)
    backwards_2 = LSTM(LSTM_output_size, go_backwards=True, name='backwards_2')(sequence2)
    after_dp_backward_2 = Dropout(0.20, name='after_dp_backward_2')(backwards_2)
    merged_2 = keras.layers.concatenate([after_dp_forward_2, after_dp_backward_2], axis=-1)
    after_dp_2 = Dropout(0.5, name='after_dp_2')(merged_2)

    input3 = Input(shape=(max_sentence_len3, embed_size_word2vec), dtype='float32')
    sequence3 = Masking(mask_value = 0.0)(input3)
    forwards_3 = LSTM(LSTM_output_size, name='forwards_3')(sequence3)
    after_dp_forward_3 = Dropout(0.20, name='after_dp_forward_3')(forwards_3)
    backwards_3 = LSTM(LSTM_output_size, go_backwards=True, name='backwards_3')(sequence3)
    after_dp_backward_3 = Dropout(0.20, name='after_dp_backward_3')(backwards_3)
    merged_3 = keras.layers.concatenate([after_dp_forward_3, after_dp_backward_3], axis=-1)
    after_dp_3 = Dropout(0.5, name='after_dp_3')(merged_3)

    merge1 = keras.layers.concatenate([after_dp_1, after_dp_2], axis=-1)
    last_merge = keras.layers.concatenate([merge1, after_dp_3], axis=-1)

    layer0 = Dense(384, activation='relu')(last_merge)
    layer1 = Dense(512, activation='relu')(layer0)
    layer2 = Dense(256, activation='relu')(layer1)
    layer3 = Dense(128, activation='relu')(layer2)
    layer4 = Dense(64, activation='relu')(layer3)
    output = Dense(2, activation='softmax', name='output')(layer4)

    model = Model(input=[input1, input2, input3], output=output) 
    rms = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08)
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    return model

def merge_4_bidirectional_RNN(max_sentence_len,max_sentence_len2, max_sentence_len3, max_sentence_len4, embed_size_word2vec, LSTM_output_size):
    input1 = Input(shape=(max_sentence_len, embed_size_word2vec), dtype='float32')
    sequence = Masking(mask_value = 0.0)(input1)
    forwards_1 = LSTM(LSTM_output_size, name='forwards_1')(sequence)
    after_dp_forward_1 = Dropout(0.20, name='after_dp_forward_1')(forwards_1) 
    backwards_1 = LSTM(LSTM_output_size, go_backwards=True, name='backwords_1')(sequence)
    after_dp_backward_1 = Dropout(0.20, name='after_dp_backward_1')(backwards_1)         
    merged_1 = keras.layers.concatenate([after_dp_forward_1, after_dp_backward_1], axis=-1)
    after_dp_1= Dropout(0.5, name='after_dp_1')(merged_1)
                
    input2 = Input(shape=(max_sentence_len2, embed_size_word2vec), dtype='float32')
    sequence2 = Masking(mask_value = 0.0)(input2)
    forwards_2 = LSTM(LSTM_output_size, name='forwards_2')(sequence2)
    after_dp_forward_2 = Dropout(0.20, name='after_dp_forward_2')(forwards_2)
    backwards_2 = LSTM(LSTM_output_size, go_backwards=True, name='backwards_2')(sequence2)
    after_dp_backward_2 = Dropout(0.20, name='after_dp_backward_2')(backwards_2)
    merged_2 = keras.layers.concatenate([after_dp_forward_2, after_dp_backward_2], axis=-1)
    after_dp_2 = Dropout(0.5, name='after_dp_2')(merged_2)

    input3 = Input(shape=(max_sentence_len3, embed_size_word2vec), dtype='float32')
    sequence3 = Masking(mask_value = 0.0)(input3)
    forwards_3 = LSTM(LSTM_output_size, name='forwards_3')(sequence3)
    after_dp_forward_3 = Dropout(0.20, name='after_dp_forward_3')(forwards_3)
    backwards_3 = LSTM(LSTM_output_size, go_backwards=True, name='backwards_3')(sequence3)
    after_dp_backward_3 = Dropout(0.20, name='after_dp_backward_3')(backwards_3)
    merged_3 = keras.layers.concatenate([after_dp_forward_3, after_dp_backward_3], axis=-1)
    after_dp_3 = Dropout(0.5, name='after_dp_3')(merged_3)

    input4 = Input(shape=(max_sentence_len4, embed_size_word2vec), dtype='float32')
    sequence4 = Masking(mask_value = 0.0)(input4)
    forwards_4 = LSTM(LSTM_output_size, name='forwards_4')(sequence4)
    after_dp_forward_4 = Dropout(0.20, name='after_dp_forward_4')(forwards_4)
    backwards_4 = LSTM(LSTM_output_size, go_backwards=True, name='backwards_4')(sequence4)
    after_dp_backward_4 = Dropout(0.20, name='after_dp_backward_4')(backwards_4)
    merged_4 = keras.layers.concatenate([after_dp_forward_4, after_dp_backward_4], axis=-1)
    after_dp_4 = Dropout(0.5, name='after_dp_4')(merged_4)

    merge1 = keras.layers.concatenate([after_dp_1, after_dp_2], axis=-1)
    merge2 = keras.layers.concatenate([merge1, after_dp_3], axis=-1)
    last_merge = keras.layers.concatenate([merge2, after_dp_4], axis=-1)

    layer0 = Dense(512, activation='relu')(last_merge)
    layer1 = Dense(768, activation='relu')(layer0)
    layer2 = Dense(512, activation='relu')(layer1)
    layer3 = Dense(256, activation='relu')(layer2)
    layer4 = Dense(128, activation='relu')(layer3)
    output = Dense(2, activation='softmax', name='output')(layer4)

    model = Model(input=[input1, input2, input3, input4], output=output) 
    rms = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08)
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    return model

def fit(model, X_train, Y_train, batch_size, epoch_num):
    hist = model.fit(X_train, Y_train, batch_size = batch_size, epochs = epoch_num, verbose = 1)
    return hist

def save_model(model, model_name, weight_name):
    model_json = model.to_json()
    with open(model_name,"w") as json_file :
        json_file.write(model_json)
    model.save_weights(weight_name)
    print("Saved model to disk\n\n")
