#!/bin/bash

#If test_rate changed, test_limit_zero and test_limit_one should be changed.
test_rate=10
#If balance = False, test all data
#If balance = True, test_limit_zero + test_limit_one
balance=False
model='glassfish'
test1='guava'
test2='hbase'

python3 load_lower_split_model.py ${test1} code ${model} code_lowercase_split ${test_rate} ${balance} 1202 1203 &> cross-project/${test_rate}/load-${model}-code_lowercase_split-test-${test1}-code_lowercase_split_alldata.txt
python3 load_lower_split_model.py ${test2} code ${model} code_lowercase_split ${test_rate} ${balance} 1202 1203 &> cross-project/${test_rate}/load-${model}-code_lowercase_split-test-${test2}-code_lowercase_split_alldata.txt
:<<END
python3 load_lower_split_model.py ${test1} AST ${model} AST_lowercase_split ${test_rate} ${balance} 1202 1203 &> cross-project/${test_rate}/load-${model}-AST_lowercase_split-test-${test1}-AST_lowercase_split_alldata.txt
python3 load_lower_split_model.py ${test2} AST ${model} AST_lowercase_split ${test_rate} ${balance} 1202 1203 &> cross-project/${test_rate}/load-${model}-AST_lowercase_split-test-${test2}-AST_lowercase_split_alldata.txt

python3 load_lower_split_model.py ${test1} CAST ${model} CAST_lowercase_split ${test_rate} ${balance} 1202 1203 &> cross-project/${test_rate}/load-${model}-CAST_lowercase_split-test-${test1}-CAST_lowercase_split_alldata.txt
python3 load_lower_split_model.py ${test2} CAST ${model} CAST_lowercase_split ${test_rate} ${balance} 1202 1203 &> cross-project/${test_rate}/load-${model}-CAST_lowercase_split-test-${test2}-CAST_lowercase_split_alldata.txt

python3 load_lower_split_model.py ${test1} CAST_s ${model} CAST_s_lowercase_split ${test_rate} ${balance} 1202 1203 &> cross-project/${test_rate}/load-${model}-CAST_s_lowercase_split-test-${test1}-CAST_s_lowercase_split_alldata.txt
python3 load_lower_split_model.py ${test2} CAST_s ${model} CAST_s_lowercase_split ${test_rate} ${balance} 1202 1203 &> cross-project/${test_rate}/load-${model}-CAST_s_lowercase_split-test-${test2}-CAST_s_lowercase_split_alldata.txt
END
