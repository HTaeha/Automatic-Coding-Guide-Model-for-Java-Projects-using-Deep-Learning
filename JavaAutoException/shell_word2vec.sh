#!/bin/bash
:<<END

input_data=guava
test_rate=1
#if first execution 1, else 0
first_execute=1

min=0
max=0
min_word_frequency=100
python3 merge_network_input4_new.py ${input_data} code 1 code ${test_rate} ${min_word_frequency}  ${first_execute} ${min} ${max}
python3 merge_network_input4_new.py ${input_data} code 1 AST ${test_rate} ${min_word_frequency}  ${first_execute} ${min} ${max}
python3 merge_network_input4_new.py ${input_data} code 1 CAST ${test_rate} ${min_word_frequency}  ${first_execute} ${min} ${max}
python3 merge_network_input4_new.py ${input_data} code 4 code AST CAST depth_num ${test_rate} ${min_word_frequency}  ${first_execute} ${min} ${max}
python3 merge_network_input4_new.py ${input_data} code 3 code AST CAST ${test_rate} ${min_word_frequency}  ${first_execute} ${min} ${max}
python3 merge_network_input4_new.py ${input_data} code 2 code AST ${test_rate} ${min_word_frequency}  ${first_execute} ${min} ${max}
input_data=hbase
test_rate=1
#if first execution 1, else 0
first_execute=1

min=0
max=0
min_word_frequency=100
python3 merge_network_input4_new.py ${input_data} code 1 code ${test_rate} ${min_word_frequency}  ${first_execute} ${min} ${max}
python3 merge_network_input4_new.py ${input_data} code 1 AST ${test_rate} ${min_word_frequency}  ${first_execute} ${min} ${max}
python3 merge_network_input4_new.py ${input_data} code 1 CAST ${test_rate} ${min_word_frequency}  ${first_execute} ${min} ${max}
python3 merge_network_input4_new.py ${input_data} code 4 code AST CAST depth_num ${test_rate} ${min_word_frequency}  ${first_execute} ${min} ${max}
python3 merge_network_input4_new.py ${input_data} code 3 code AST CAST ${test_rate} ${min_word_frequency}  ${first_execute} ${min} ${max}
python3 merge_network_input4_new.py ${input_data} code 2 code AST ${test_rate} ${min_word_frequency}  ${first_execute} ${min} ${max}
END
input_data=glassfish
test_rate=1
#if first execution 1, else 0
first_execute=1

min=0
max=0
min_word_frequency=100
python3 merge_network_input4_new.py ${input_data} code 1 code ${test_rate} ${min_word_frequency}  ${first_execute} ${min} ${max}
python3 merge_network_input4_new.py ${input_data} code 1 AST ${test_rate} ${min_word_frequency}  ${first_execute} ${min} ${max}
python3 merge_network_input4_new.py ${input_data} code 1 CAST ${test_rate} ${min_word_frequency}  ${first_execute} ${min} ${max}
python3 merge_network_input4_new.py ${input_data} code 4 code AST CAST depth_num ${test_rate} ${min_word_frequency}  ${first_execute} ${min} ${max}
python3 merge_network_input4_new.py ${input_data} code 3 code AST CAST ${test_rate} ${min_word_frequency}  ${first_execute} ${min} ${max}
python3 merge_network_input4_new.py ${input_data} code 2 code AST ${test_rate} ${min_word_frequency}  ${first_execute} ${min} ${max}
:<<END
END
