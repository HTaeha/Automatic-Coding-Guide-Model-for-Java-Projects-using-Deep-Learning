#!/bin/bash

input_data=guava

min_word_frequency=100
python3 ../src/make_wordvec_model.py ${input_data} 1 code ${min_word_frequency}
python3 ../src/make_wordvec_model.py ${input_data} 1 AST ${min_word_frequency}
python3 ../src/make_wordvec_model.py ${input_data} 1 CAST ${min_word_frequency}
python3 ../src/make_wordvec_model.py ${input_data} 4 code AST CAST depth_num ${min_word_frequency}
python3 ../src/make_wordvec_model.py ${input_data} 3 code AST CAST ${min_word_frequency}
python3 ../src/make_wordvec_model.py ${input_data} 2 code AST ${min_word_frequency}

