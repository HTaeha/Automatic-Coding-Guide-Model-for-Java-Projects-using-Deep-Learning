#!/bin/bash
input_data=hbase
test_rate=1

min=1
max=300
min_word_frequency=0
#python3 sentence_ensemble_bagging.py ${input_data} 1 code ${test_rate} ${min_word_frequency}  ${min} ${max} &> Ensemble/${test_rate}/${input_data}--code_sentence_balanced_min${min}_max${max}_frequency${min_word_frequency}.txt
#python3 sentence_ensemble_bagging.py ${input_data} 1 AST ${test_rate} ${min_word_frequency}  ${min} ${max} &> Ensemble/${test_rate}/${input_data}--AST_sentence_balanced_min${min}_max${max}_frequency${min_word_frequency}.txt
#python3 sentence_ensemble_bagging.py ${input_data} 1 CAST ${test_rate} ${min_word_frequency}  ${min} ${max} &> Ensemble/${test_rate}/${input_data}--CAST_sentence_balanced_min${min}_max${max}_frequency${min_word_frequency}.txt
#python3 sentence_ensemble_bagging.py ${input_data} 4 code AST CAST depth_num ${test_rate} ${min_word_frequency}  ${min} ${max}&> Ensemble/${test_rate}/${input_data}--CAST_m_cACd_sentence_balanced_min${min}_max${max}_frequency${min_word_frequency}.txt
python3 sentence_ensemble_bagging.py ${input_data} 3 code AST CAST ${test_rate} ${min_word_frequency}  ${min} ${max}&> Ensemble/${test_rate}/${input_data}-code,AST,CAST_sentence_balanced_min${min}_max${max}_frequency${min_word_frequency}.txt
python3 sentence_ensemble_bagging.py ${input_data} 2 code AST ${test_rate} ${min_word_frequency}  ${min} ${max}&> Ensemble/${test_rate}/${input_data}-code,AST_sentence_balanced_min${min}_max${max}_frequency${min_word_frequency}.txt
python3 sentence_ensemble_bagging.py ${input_data} 2 code CAST ${test_rate} ${min_word_frequency}  ${min} ${max}&> Ensemble/${test_rate}/${input_data}-code,CAST_sentence_balanced_min${min}_max${max}_frequency${min_word_frequency}.txt
python3 sentence_ensemble_bagging.py ${input_data} 2 AST CAST ${test_rate} ${min_word_frequency}  ${min} ${max}&> Ensemble/${test_rate}/${input_data}-AST,CAST_sentence_balanced_min${min}_max${max}_frequency${min_word_frequency}.txt


