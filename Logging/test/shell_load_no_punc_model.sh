#!/bin/bash

#If test_rate changed, test_limit_zero and test_limit_one should be changed.
test_rate=1
#If balance = False, test all data
#If balance = True, test_limit_zero + test_limit_one
balance=False
model='glassfish'
test1='guava'
test2='hbase'
frequency=10000

model='hbase'
test1='guava'
test2='glassfish'
frequency=500

:<<'END'
END
python3 load_no_punc_model.py ${test1} code ${model} code ${frequency} ${test_rate} ${balance}   &> cross-project/real_final/${test_rate}/load-${model}-code_no_punc_frequency${frequency}_LSTM64_batch64-test-${test1}-code_no_punc.txt
python3 load_no_punc_model.py ${test2} code ${model} code ${frequency} ${test_rate} ${balance}   &> cross-project/real_final/${test_rate}/load-${model}-code_no_punc_frequency${frequency}_LSTM64_batch64-test-${test2}-code_no_punc.txt

#python3 load_no_punc_model.py ${test1} AST ${model} AST ${frequency} ${test_rate} ${balance}   &> cross-project/real_final/${test_rate}/load-${model}-AST_no_punc_frequency${frequency}_LSTM64_batch64-test-${test1}-AST_no_punc.txt
#python3 load_no_punc_model.py ${test2} AST ${model} AST ${frequency} ${test_rate} ${balance}   &> cross-project/real_final/${test_rate}/load-${model}-AST_no_punc_frequency${frequency}_LSTM64_batch64-test-${test2}-AST_no_punc.txt

python3 load_no_punc_model.py ${test1} CAST ${model} CAST ${frequency} ${test_rate} ${balance}   &> cross-project/real_final/${test_rate}/load-${model}-CAST_no_punc_frequency${frequency}_LSTM64_batch64-test-${test1}-CAST_no_punc.txt
python3 load_no_punc_model.py ${test2} CAST ${model} CAST ${frequency} ${test_rate} ${balance}   &> cross-project/real_final/${test_rate}/load-${model}-CAST_no_punc_frequency${frequency}_LSTM64_batch64-test-${test2}-CAST_no_punc.txt

python3 load_no_punc_model.py ${test1} CAST_s ${model} CAST_s ${frequency} ${test_rate} ${balance}   &> cross-project/real_final/${test_rate}/load-${model}-CAST_s_no_punc_frequency${frequency}_LSTM64_batch64-test-${test1}-CAST_s_no_punc.txt
python3 load_no_punc_model.py ${test2} CAST_s ${model} CAST_s ${frequency} ${test_rate} ${balance}   &> cross-project/real_final/${test_rate}/load-${model}-CAST_s_no_punc_frequency${frequency}_LSTM64_batch64-test-${test2}-CAST_s_no_punc.txt

python3 load_no_punc_merge_network_2_wordvec_model.py ${test1} ${model} ${frequency} ${test_rate} ${balance}   &> cross-project/real_final/${test_rate}/load-${model}-CAST_m_no_punc_frequency${frequency}_LSTM64_batch64-test-${test1}-CAST_m_no_punc.txt
python3 load_no_punc_merge_network_2_wordvec_model.py ${test2} ${model} ${frequency} ${test_rate} ${balance}   &> cross-project/real_final/${test_rate}/load-${model}-CAST_m_no_punc_frequency${frequency}_LSTM64_batch64-test-${test2}-CAST_m_no_punc.txt

python3 load_no_punc_doc_product_model.py ${test1} ${model} ${frequency} ${test_rate} ${balance}   &> cross-project/real_final/${test_rate}/load-${model}-CAST_d_no_punc_frequency${frequency}_LSTM64_batch64-test-${test1}-CAST_d_no_punc.txt
python3 load_no_punc_doc_product_model.py ${test2} ${model} ${frequency} ${test_rate} ${balance}   &> cross-project/real_final/${test_rate}/load-${model}-CAST_d_no_punc_frequency${frequency}_LSTM64_batch64-test-${test2}-CAST_d_no_punc.txt

model='guava'
test1='glassfish'
test2='hbase'
frequency=10000

