#!/bin/bash
input_data=guava
test_rate=1
#if first execution 1, else 0
first_execute=0

min=1
max=200
min_word_frequency=100
train_num=5000
python3 sentence_java_auto_logging.py ${input_data} 1 code ${test_rate} ${min_word_frequency}  ${first_execute} ${min} ${max} ${train_num} &> CV/real_last/${test_rate}/ETE-${input_data}-code_sentence_balanced_min${min}_max${max}_frequency${min_word_frequency}_trainNum${train_num}.txt
python3 sentence_java_auto_logging.py ${input_data} 1 AST ${test_rate} ${min_word_frequency}  ${first_execute} ${min} ${max} ${train_num} &> CV/real_last/${test_rate}/ETE-${input_data}-AST_sentence_balanced_min${min}_max${max}_frequency${min_word_frequency}_trainNum${train_num}.txt
python3 sentence_java_auto_logging.py ${input_data} 1 CAST ${test_rate} ${min_word_frequency}  ${first_execute} ${min} ${max} ${train_num} &> CV/real_last/${test_rate}/ETE-${input_data}-CAST_sentence_balanced_min${min}_max${max}_frequency${min_word_frequency}_trainNum${train_num}.txt
python3 sentence_java_auto_logging.py ${input_data} 4 code AST CAST depth_num ${test_rate} ${min_word_frequency}  ${first_execute} ${min} ${max} ${train_num} &> CV/real_last/${test_rate}/ETE-${input_data}-CAST_m_cACd_sentence_balanced_min${min}_max${max}_frequency${min_word_frequency}_trainNum${train_num}.txt
python3 sentence_java_auto_logging.py ${input_data} 3 code AST CAST ${test_rate} ${min_word_frequency}  ${first_execute} ${min} ${max} ${train_num} &> CV/real_last/${test_rate}/ETE-${input_data}-CAST_m_cAC_sentence_balanced_min${min}_max${max}_frequency${min_word_frequency}_trainNum${train_num}.txt
python3 sentence_java_auto_logging.py ${input_data} 2 code AST ${test_rate} ${min_word_frequency}  ${first_execute} ${min} ${max} ${train_num} &> CV/real_last/${test_rate}/ETE-${input_data}-CAST_m_cA_sentence_balanced_min${min}_max${max}_frequency${min_word_frequency}_trainNum${train_num}.txt

