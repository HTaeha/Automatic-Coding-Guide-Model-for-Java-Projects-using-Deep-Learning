#!/bin/bash
input_data=glassfish
test_rate=1
nth_fold=5
seed=1398

min=1
max=200
min_word_frequency=100
label=Exception
#python3 ../src/train.py ${input_data} 1 code ${test_rate} ${min_word_frequency}  ${seed} ${min} ${max} ${label} ${nth_fold} &> ../${label}/Result/${test_rate}/${input_data}-code_min${min}_max${max}_frequency${min_word_frequency}.txt
#python3 ../src/train.py ${input_data} 1 AST ${test_rate} ${min_word_frequency}  ${seed} ${min} ${max} ${label} ${nth_fold} &> ../${label}/Result/${test_rate}/${input_data}-AST_min${min}_max${max}_frequency${min_word_frequency}.txt
#python3 ../src/train.py ${input_data} 1 CAST ${test_rate} ${min_word_frequency}  ${seed} ${min} ${max} ${label} ${nth_fold} &> ../${label}/Result/${test_rate}/${input_data}-CAST_min${min}_max${max}_frequency${min_word_frequency}.txt
python3 ../src/train.py ${input_data} 4 code AST CAST depth_num ${test_rate} ${min_word_frequency}  ${seed} ${min} ${max} ${label} ${nth_fold} &> ../${label}/Result/${test_rate}/${input_data}-CAST_m_cACd_min${min}_max${max}_frequency${min_word_frequency}.txt
python3 ../src/train.py ${input_data} 3 code AST CAST ${test_rate} ${min_word_frequency}  ${seed} ${min} ${max} ${label} ${nth_fold} &> ../${label}/Result/${test_rate}/${input_data}-CAST_m_cAC_min${min}_max${max}_frequency${min_word_frequency}.txt
#python3 ../src/train.py ${input_data} 2 code AST ${test_rate} ${min_word_frequency}  ${seed} ${min} ${max} ${label} ${nth_fold} &> ../${label}/Result/${test_rate}/${input_data}-CAST_m_cA_min${min}_max${max}_frequency${min_word_frequency}.txt

