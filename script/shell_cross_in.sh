#!/bin/bash
target_data=hbase
source_data=guava
test_rate=1


min=1
max=200
min_word_frequency=100
python3 load_model_sentence_len.py ${target_data} ${source_data} code 1 code ${test_rate} ${min_word_frequency}  ${min} ${max} &> cross-project/end2/${test_rate}/${target_data}-${source_data}-code_sentence_balanced_min${min}_max${max}_frequency${min_word_frequency}.txt
python3 load_model_sentence_len.py ${target_data} ${source_data} AST 1 AST ${test_rate} ${min_word_frequency}  ${min} ${max} &> cross-project/end2/${test_rate}/${target_data}-${source_data}-AST_sentence_balanced_min${min}_max${max}_frequency${min_word_frequency}.txt
python3 load_model_sentence_len.py ${target_data} ${source_data} CAST 1 CAST ${test_rate} ${min_word_frequency}  ${min} ${max} &> cross-project/end2/${test_rate}/${target_data}-${source_data}-CAST_sentence_balanced_min${min}_max${max}_frequency${min_word_frequency}.txt
python3 load_model_sentence_len.py ${target_data} ${source_data} CAST_m_cACd 4 code AST CAST depth_num ${test_rate} ${min_word_frequency}  ${min} ${max}&> cross-project/end2/${test_rate}/${target_data}-${source_data}-CAST_m_cACd_sentence_balanced_min${min}_max${max}_frequency${min_word_frequency}.txt
python3 load_model_sentence_len.py ${target_data} ${source_data} CAST_m_cAC 3 code AST CAST ${test_rate} ${min_word_frequency}  ${min} ${max}&> cross-project/end2/${test_rate}/${target_data}-${source_data}-CAST_m_cAC_sentence_balanced_min${min}_max${max}_frequency${min_word_frequency}.txt
python3 load_model_sentence_len.py ${target_data} ${source_data} CAST_m_cA 2 code AST ${test_rate} ${min_word_frequency}  ${min} ${max}&> cross-project/end2/${test_rate}/${target_data}-${source_data}-CAST_m_cA_sentence_balanced_min${min}_max${max}_frequency${min_word_frequency}.txt
:<<END
python3 load_model_test_all.py ${target_data} ${source_data} code 1 code ${test_rate} ${min_word_frequency}  ${min} ${max} &> cross-project/end2/${test_rate}/test_all-${target_data}-${source_data}-code_sentence_balanced_min${min}_max${max}_frequency${min_word_frequency}.txt
python3 load_model_test_all.py ${target_data} ${source_data} AST 1 AST ${test_rate} ${min_word_frequency}  ${min} ${max} &> cross-project/end2/${test_rate}/test_all-${target_data}-${source_data}-AST_sentence_balanced_min${min}_max${max}_frequency${min_word_frequency}.txt
python3 load_model_test_all.py ${target_data} ${source_data} CAST 1 CAST ${test_rate} ${min_word_frequency}  ${min} ${max} &> cross-project/end2/${test_rate}/test_all-${target_data}-${source_data}-CAST_sentence_balanced_min${min}_max${max}_frequency${min_word_frequency}.txt
python3 load_model_test_all.py ${target_data} ${source_data} CAST_m_cACd 4 code AST CAST depth_num ${test_rate} ${min_word_frequency}  ${min} ${max}&> cross-project/end2/${test_rate}/test_all-${target_data}-${source_data}-CAST_m_cACd_sentence_balanced_min${min}_max${max}_frequency${min_word_frequency}.txt
python3 load_model_test_all.py ${target_data} ${source_data} CAST_m_cAC 3 code AST CAST ${test_rate} ${min_word_frequency}  ${min} ${max}&> cross-project/end2/${test_rate}/test_all-${target_data}-${source_data}-CAST_m_cAC_sentence_balanced_min${min}_max${max}_frequency${min_word_frequency}.txt
python3 load_model_test_all.py ${target_data} ${source_data} CAST_m_cA 2 code AST ${test_rate} ${min_word_frequency}  ${min} ${max}&> cross-project/end2/${test_rate}/test_all-${target_data}-${source_data}-CAST_m_cA_sentence_balanced_min${min}_max${max}_frequency${min_word_frequency}.txt

END


