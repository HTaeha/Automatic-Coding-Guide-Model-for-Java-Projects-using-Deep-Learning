#!/bin/bash
test_rate=1
min_word_frequency=500
LSTM=64
:<<END
END
python3 no_punc_balanced_auto_exception.py hbase code ${test_rate} ${min_word_frequency} ${LSTM}&> CV/real_final/${test_rate}/hbase-code_no_punc_frequency${min_word_frequency}_LSTM${LSTM}_batch64.txt
python3 no_punc_balanced_auto_exception.py hbase AST ${test_rate} ${min_word_frequency} ${LSTM}&> CV/real_final/${test_rate}/hbase-AST_no_punc_frequency${min_word_frequency}_LSTM${LSTM}_batch64.txt
python3 no_punc_balanced_auto_exception.py hbase CAST ${test_rate} ${min_word_frequency} ${LSTM}&> CV/real_final/${test_rate}/hbase-CAST_no_punc_frequency${min_word_frequency}_LSTM${LSTM}_batch64.txt

python3 no_punc_balanced_auto_exception.py hbase CAST_s ${test_rate} ${min_word_frequency} ${LSTM}&> CV/real_final/${test_rate}/hbase-CAST_s_no_punc_frequency${min_word_frequency}_LSTM${LSTM}_batch64.txt
python3 no_punc_merge_network_2_wordvec_model.py hbase ${test_rate} ${min_word_frequency} ${LSTM}&> CV/real_final/${test_rate}/hbase-CAST_m_no_punc_frequency${min_word_frequency}_LSTM${LSTM}_batch64.txt
python3 no_punc_doc_product.py hbase ${test_rate} ${min_word_frequency} ${LSTM}&> CV/real_final/${test_rate}/hbase-CAST_d_no_punc_frequency${min_word_frequency}_LSTM${LSTM}_batch64.txt

:<<END
min_word_frequency=5000
LSTM=64
python3 no_punc_balanced_auto_exception.py hbase code ${test_rate} ${min_word_frequency} ${LSTM}&> CV/real_final/${test_rate}/hbase-code_no_punc_frequency${min_word_frequency}_LSTM${LSTM}_batch64.txt
python3 no_punc_balanced_auto_exception.py hbase AST ${test_rate} ${min_word_frequency} ${LSTM}&> CV/real_final/${test_rate}/hbase-AST_no_punc_frequency${min_word_frequency}_LSTM${LSTM}_batch64.txt
python3 no_punc_balanced_auto_exception.py hbase CAST ${test_rate} ${min_word_frequency} ${LSTM}&> CV/real_final/${test_rate}/hbase-CAST_no_punc_frequency${min_word_frequency}_LSTM${LSTM}_batch64.txt

python3 no_punc_balanced_auto_exception.py hbase CAST_s ${test_rate} ${min_word_frequency} ${LSTM}&> CV/real_final/${test_rate}/hbase-CAST_s_no_punc_frequency${min_word_frequency}_LSTM${LSTM}_batch64.txt
python3 no_punc_merge_network_2_wordvec_model.py hbase ${test_rate} ${min_word_frequency} ${LSTM}&> CV/real_final/${test_rate}/hbase-CAST_m_no_punc_frequency${min_word_frequency}_LSTM${LSTM}_batch64.txt
python3 no_punc_doc_product.py hbase ${test_rate} ${min_word_frequency} ${LSTM}&> CV/real_final/${test_rate}/hbase-CAST_d_no_punc_frequency${min_word_frequency}_LSTM${LSTM}_batch64.txt
END

:<<END
python3 no_punc_balanced_auto_exception.py hbase code ${test_rate} ${min_word_frequency}&> CV/real_final/${test_rate}/hbase-code_frequency${min_word_frequency}_LSTM32_batch64.txt
#python3 no_punc_balanced_auto_exception.py guava code ${test_rate} ${min_word_frequency}&> CV/real_final/${test_rate}/guava-code_frequency${min_word_frequency}.txt
#python3 no_punc_balanced_auto_exception.py glassfish code ${test_rate} ${min_word_frequency}&> CV/real_final/${test_rate}/glassfish-code_frequency${min_word_frequency}.txt

#python3 no_punc_balanced_auto_exception.py guava AST ${test_rate} ${min_word_frequency}&> CV/real_final/${test_rate}/guava-AST_frequency${min_word_frequency}.txt
#python3 no_punc_balanced_auto_exception.py glassfish AST ${test_rate} ${min_word_frequency}&> CV/real_final/${test_rate}/glassfish-AST_frequency${min_word_frequency}.txt
python3 no_punc_balanced_auto_exception.py hbase AST ${test_rate} ${min_word_frequency}&> CV/real_final/${test_rate}/hbase-AST_frequency${min_word_frequency}_LSTM32_batch64.txt

python3 no_punc_balanced_auto_exception.py hbase CAST ${test_rate} ${min_word_frequency}&> CV/real_final/${test_rate}/hbase-CAST_frequency${min_word_frequency}_LSTM32_batch64.txt
END

:<<END
python3 no_punc_balanced_auto_exception.py hbase CAST_s ${test_rate} &> CV/real_final/${test_rate}/hbase-CAST_s.txt
python3 no_punc_balanced_auto_exception.py guava CAST_s ${test_rate} &> CV/real_final/${test_rate}/guava-CAST_s.txt
python3 no_punc_balanced_auto_exception.py glassfish CAST_s ${test_rate} &> CV/real_final/${test_rate}/glassfish-CAST_s.txt

python3 no_punc_merge_network_2_wordvec_model.py guava ${test_rate} &> CV/real_final/${test_rate}/guava-CAST_m.txt
python3 no_punc_merge_network_2_wordvec_model.py hbase ${test_rate} &> CV/real_final/${test_rate}/hbase-CAST_m.txt
python3 no_punc_merge_network_2_wordvec_model.py glassfish ${test_rate} &> CV/real_final/${test_rate}/glassfish-CAST_m.txt

python3 no_punc_doc_product.py guava ${test_rate} &> CV/real_final/${test_rate}/guava-CAST_d.txt
python3 no_punc_doc_product.py hbase ${test_rate} &> CV/real_final/${test_rate}/hbase-CAST_d.txt
python3 no_punc_doc_product.py glassfish ${test_rate} &> CV/real_final/${test_rate}/glassfish-CAST_d.txt
END
