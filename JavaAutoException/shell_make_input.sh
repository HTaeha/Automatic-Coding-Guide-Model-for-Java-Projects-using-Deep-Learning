#!/bin/bash
:<<END
input_data=glassfish
test_rate=1
#if first execution 1, else 0
first_execute=0

min=1
max=400
min_word_frequency=100
python3 make_sentence_balanced_input_data.py ${input_data} code 4 code AST CAST depth_num ${test_rate} ${min_word_frequency}  ${first_execute} ${min} ${max}
input_data=hbase
test_rate=1
#if first execution 1, else 0
first_execute=0

min=1
max=400
min_word_frequency=100
python3 make_sentence_balanced_input_data.py ${input_data} code 4 code AST CAST depth_num ${test_rate} ${min_word_frequency}  ${first_execute} ${min} ${max}
END
input_data=guava
test_rate=1
#if first execution 1, else 0
first_execute=0

min=1
max=400
min_word_frequency=100
python3 make_sentence_balanced_input_data.py ${input_data} code 4 code AST CAST depth_num ${test_rate} ${min_word_frequency}  ${first_execute} ${min} ${max}
:<<END
END
