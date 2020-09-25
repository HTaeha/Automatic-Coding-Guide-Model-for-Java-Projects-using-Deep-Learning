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
frequency=5000

:<<'END'
END
python3 load_model.py ${test1} code ${model} code ${frequency} ${test_rate} ${balance}   &> cross-project/real_final/${test_rate}/load-${model}-code_combine_wordvec_frequency${frequency}_LSTM64_batch64-test-${test1}-code.txt
python3 load_model.py ${test2} code ${model} code ${frequency} ${test_rate} ${balance}   &> cross-project/real_final/${test_rate}/load-${model}-code_combine_wordvec_frequency${frequency}_LSTM64_batch64-test-${test2}-code.txt

python3 load_model.py ${test1} AST ${model} AST ${frequency} ${test_rate} ${balance}   &> cross-project/real_final/${test_rate}/load-${model}-AST_combine_wordvec_frequency${frequency}_LSTM64_batch64-test-${test1}-AST.txt
python3 load_model.py ${test2} AST ${model} AST ${frequency} ${test_rate} ${balance}   &> cross-project/real_final/${test_rate}/load-${model}-AST_combine_wordvec_frequency${frequency}_LSTM64_batch64-test-${test2}-AST.txt

python3 load_model.py ${test1} CAST ${model} CAST ${frequency} ${test_rate} ${balance}   &> cross-project/real_final/${test_rate}/load-${model}-CAST_combine_wordvec_frequency${frequency}_LSTM64_batch64-test-${test1}-CAST.txt
python3 load_model.py ${test2} CAST ${model} CAST ${frequency} ${test_rate} ${balance}   &> cross-project/real_final/${test_rate}/load-${model}-CAST_combine_wordvec_frequency${frequency}_LSTM64_batch64-test-${test2}-CAST.txt

python3 load_model.py ${test1} CAST_s ${model} CAST_s ${frequency} ${test_rate} ${balance}   &> cross-project/real_final/${test_rate}/load-${model}-CAST_s_combine_wordvec_frequency${frequency}_LSTM64_batch64-test-${test1}-CAST_s.txt
python3 load_model.py ${test2} CAST_s ${model} CAST_s ${frequency} ${test_rate} ${balance}   &> cross-project/real_final/${test_rate}/load-${model}-CAST_s_combine_wordvec_frequency${frequency}_LSTM64_batch64-test-${test2}-CAST_s.txt

python3 load_merge_network.py ${test1} ${model} ${frequency} ${test_rate} ${balance}   &> cross-project/real_final/${test_rate}/load-${model}-CAST_m_combine_wordvec_frequency${frequency}_LSTM64_batch64-test-${test1}-CAST_m.txt
python3 load_merge_network.py ${test2} ${model} ${frequency} ${test_rate} ${balance}   &> cross-project/real_final/${test_rate}/load-${model}-CAST_m_combine_wordvec_frequency${frequency}_LSTM64_batch64-test-${test2}-CAST_m.txt

python3 load_doc_product_model.py ${test1} ${model} ${frequency} ${test_rate} ${balance}   &> cross-project/real_final/${test_rate}/load-${model}-CAST_d_combine_wordvec_frequency${frequency}_LSTM64_batch64-test-${test1}-CAST_d.txt
python3 load_doc_product_model.py ${test2} ${model} ${frequency} ${test_rate} ${balance}   &> cross-project/real_final/${test_rate}/load-${model}-CAST_d_combine_wordvec_frequency${frequency}_LSTM64_batch64-test-${test2}-CAST_d.txt
frequency=500

:<<'END'
END
python3 load_model.py ${test1} code ${model} code ${frequency} ${test_rate} ${balance}   &> cross-project/real_final/${test_rate}/load-${model}-code_combine_wordvec_frequency${frequency}_LSTM64_batch64-test-${test1}-code.txt
python3 load_model.py ${test2} code ${model} code ${frequency} ${test_rate} ${balance}   &> cross-project/real_final/${test_rate}/load-${model}-code_combine_wordvec_frequency${frequency}_LSTM64_batch64-test-${test2}-code.txt

python3 load_model.py ${test1} AST ${model} AST ${frequency} ${test_rate} ${balance}   &> cross-project/real_final/${test_rate}/load-${model}-AST_combine_wordvec_frequency${frequency}_LSTM64_batch64-test-${test1}-AST.txt
python3 load_model.py ${test2} AST ${model} AST ${frequency} ${test_rate} ${balance}   &> cross-project/real_final/${test_rate}/load-${model}-AST_combine_wordvec_frequency${frequency}_LSTM64_batch64-test-${test2}-AST.txt

python3 load_model.py ${test1} CAST ${model} CAST ${frequency} ${test_rate} ${balance}   &> cross-project/real_final/${test_rate}/load-${model}-CAST_combine_wordvec_frequency${frequency}_LSTM64_batch64-test-${test1}-CAST.txt
python3 load_model.py ${test2} CAST ${model} CAST ${frequency} ${test_rate} ${balance}   &> cross-project/real_final/${test_rate}/load-${model}-CAST_combine_wordvec_frequency${frequency}_LSTM64_batch64-test-${test2}-CAST.txt

python3 load_model.py ${test1} CAST_s ${model} CAST_s ${frequency} ${test_rate} ${balance}   &> cross-project/real_final/${test_rate}/load-${model}-CAST_s_combine_wordvec_frequency${frequency}_LSTM64_batch64-test-${test1}-CAST_s.txt
python3 load_model.py ${test2} CAST_s ${model} CAST_s ${frequency} ${test_rate} ${balance}   &> cross-project/real_final/${test_rate}/load-${model}-CAST_s_combine_wordvec_frequency${frequency}_LSTM64_batch64-test-${test2}-CAST_s.txt

python3 load_merge_network.py ${test1} ${model} ${frequency} ${test_rate} ${balance}   &> cross-project/real_final/${test_rate}/load-${model}-CAST_m_combine_wordvec_frequency${frequency}_LSTM64_batch64-test-${test1}-CAST_m.txt
python3 load_merge_network.py ${test2} ${model} ${frequency} ${test_rate} ${balance}   &> cross-project/real_final/${test_rate}/load-${model}-CAST_m_combine_wordvec_frequency${frequency}_LSTM64_batch64-test-${test2}-CAST_m.txt

python3 load_doc_product_model.py ${test1} ${model} ${frequency} ${test_rate} ${balance}   &> cross-project/real_final/${test_rate}/load-${model}-CAST_d_combine_wordvec_frequency${frequency}_LSTM64_batch64-test-${test1}-CAST_d.txt
python3 load_doc_product_model.py ${test2} ${model} ${frequency} ${test_rate} ${balance}   &> cross-project/real_final/${test_rate}/load-${model}-CAST_d_combine_wordvec_frequency${frequency}_LSTM64_batch64-test-${test2}-CAST_d.txt


model='guava'
test1='glassfish'
test2='hbase'
frequency=10000

