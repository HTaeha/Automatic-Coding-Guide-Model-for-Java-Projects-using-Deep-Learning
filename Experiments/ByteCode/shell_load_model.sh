#!/bin/bash

#If test_rate changed, test_limit_zero and test_limit_one should be changed.
test_rate=10
#If balance = False, test all data
#If balance = True, test_limit_zero + test_limit_one
balance=False
model='guava'
test1='hbase'
test2='glassfish'

python3 load_model.py ${test1} bytecode_exception ${model} bytecode_exception ${test_rate} ${balance} 1202 1203 &> cross-project/${test_rate}/load-${model}-bytecode_exception-test-${test1}-bytecode_exception.txt
:<< 'END'
python3 load_model.py ${test1} code ${model} code ${test_rate} ${balance} 1202 1203 &> cross-project/${test_rate}/load-${model}-code-test-${test1}-code.txt
python3 load_model.py ${test2} code ${model} code ${test_rate} ${balance} 1202 1203 &> cross-project/${test_rate}/load-${model}-code-test-${test2}-code.txt

python3 load_model.py ${test1} AST ${model} AST ${test_rate} ${balance} 1202 1203 &> cross-project/${test_rate}/load-${model}-AST-test-${test1}-AST.txt
python3 load_model.py ${test2} AST ${model} AST ${test_rate} ${balance} 1202 1203 &> cross-project/${test_rate}/load-${model}-AST-test-${test2}-AST.txt

python3 load_model.py ${test1} CAST ${model} CAST ${test_rate} ${balance} 1202 1203 &> cross-project/${test_rate}/load-${model}-CAST-test-${test1}-CAST.txt
python3 load_model.py ${test2} CAST ${model} CAST ${test_rate} ${balance} 1202 1203 &> cross-project/${test_rate}/load-${model}-CAST-test-${test2}-CAST.txt

python3 load_model.py ${test1} CAST_s ${model} CAST_s ${test_rate} ${balance} 1202 1203 &> cross-project/${test_rate}/load-${model}-CAST_s-test-${test1}-CAST_s.txt
python3 load_model.py ${test2} CAST_s ${model} CAST_s ${test_rate} ${balance} 1202 1203 &> cross-project/${test_rate}/load-${model}-CAST_s-test-${test2}-CAST_s.txt
END
