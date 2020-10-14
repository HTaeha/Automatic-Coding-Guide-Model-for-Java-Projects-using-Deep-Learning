#!/bin/bash
input_data=hbase
input_type=code
target_type=AST
min_word_frequency_word2vec=100
min=1
max=200

#python3 logging_code_seq2seq.py ${input_data} ${input_type} ${target_type} ${min_word_frequency_word2vec} ${min} ${max} &> Result/S2S-${input_data}_${input_type}-${target_type}_no_space_sentence_balanced_min${min}_max${max}_frequency${min_word_frequency_word2vec}.txt
python3 logging_nmt_with_attention.py ${input_data} ${input_type} ${target_type} ${min_word_frequency_word2vec} ${min} ${max} &> Result/Attention-S2S-${input_data}_${input_type}-${target_type}_sentence_balanced_min${min}_max${max}_frequency${min_word_frequency_word2vec}.txt
