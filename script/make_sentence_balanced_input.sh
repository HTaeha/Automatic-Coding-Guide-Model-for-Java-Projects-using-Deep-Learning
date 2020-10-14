#!/bin/bash

seed=1398

min=1
max=400
min_word_frequency=100

:<<END
END
input_data=hbase
python3 ../src/make_sentence_balanced_input_data.py ${input_data} code 4 code AST CAST depth_num ${min_word_frequency}  ${seed} ${min} ${max}

input_data=glassfish
python3 ../src/make_sentence_balanced_input_data.py ${input_data} code 4 code AST CAST depth_num ${min_word_frequency}  ${seed} ${min} ${max}

input_data=guava
python3 ../src/make_sentence_balanced_input_data.py ${input_data} code 4 code AST CAST depth_num ${min_word_frequency}  ${seed} ${min} ${max}
