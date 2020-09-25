#!/bin/bash

#If test_rate changed, test_limit_zero and test_limit_one should be changed.
test_rate=10
#If balance = False, test all data
#If balance = True, test_limit_zero + test_limit_one
balance=False
model='guava'
test1='hbase'
test2='glassfish'

python3 load_lower_model.py ${test1} code ${model} code_lowercase ${test_rate} ${balance} 1202 1203 &> cross-project/${test_rate}/load-${model}-code_lowercase-test-${test1}-code_lowercase_alldata.txt
python3 load_lower_model.py ${test2} code ${model} code_lowercase ${test_rate} ${balance} 1202 1203 &> cross-project/${test_rate}/load-${model}-code_lowercase-test-${test2}-code_lowercase_alldata.txt

python3 load_lower_model.py ${test1} AST ${model} AST_lowercase ${test_rate} ${balance} 1202 1203 &> cross-project/${test_rate}/load-${model}-AST_lowercase-test-${test1}-AST_lowercase_alldata.txt
python3 load_lower_model.py ${test2} AST ${model} AST_lowercase ${test_rate} ${balance} 1202 1203 &> cross-project/${test_rate}/load-${model}-AST_lowercase-test-${test2}-AST_lowercase_alldata.txt

python3 load_lower_model.py ${test1} CAST ${model} CAST_lowercase ${test_rate} ${balance} 1202 1203 &> cross-project/${test_rate}/load-${model}-CAST_lowercase-test-${test1}-CAST_lowercase_alldata.txt
python3 load_lower_model.py ${test2} CAST ${model} CAST_lowercase ${test_rate} ${balance} 1202 1203 &> cross-project/${test_rate}/load-${model}-CAST_lowercase-test-${test2}-CAST_lowercase_alldata.txt

python3 load_lower_model.py ${test1} CAST_s ${model} CAST_s_lowercase ${test_rate} ${balance} 1202 1203 &> cross-project/${test_rate}/load-${model}-CAST_s_lowercase-test-${test1}-CAST_s_lowercase_alldata.txt
python3 load_lower_model.py ${test2} CAST_s ${model} CAST_s_lowercase ${test_rate} ${balance} 1202 1203 &> cross-project/${test_rate}/load-${model}-CAST_s_lowercase-test-${test2}-CAST_s_lowercase_alldata.txt

