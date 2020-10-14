#!/bin/bash
test_rate=7

:<<'END'
END
python3 balanced_auto_exception.py hbase CAST_s ${test_rate} &> CV/real_final/${test_rate}/hbase-CAST_s.txt
python3 balanced_auto_exception.py guava CAST_s ${test_rate} &> CV/real_final/${test_rate}/guava-CAST_s.txt
python3 balanced_auto_exception.py glassfish CAST_s ${test_rate} &> CV/real_final/${test_rate}/glassfish-CAST_s.txt

python3 balanced_auto_exception.py guava AST ${test_rate} &> CV/real_final/${test_rate}/guava-AST.txt
python3 balanced_auto_exception.py glassfish AST ${test_rate} &> CV/real_final/${test_rate}/glassfish-AST.txt
python3 balanced_auto_exception.py hbase AST ${test_rate} &> CV/real_final/${test_rate}/hbase-AST.txt

python3 balanced_auto_exception.py hbase CAST ${test_rate} &> CV/real_final/${test_rate}/hbase-CAST.txt
python3 balanced_auto_exception.py guava CAST ${test_rate} &> CV/real_final/${test_rate}/guava-CAST.txt
python3 balanced_auto_exception.py glassfish CAST ${test_rate} &> CV/real_final/${test_rate}/glassfish-CAST.txt

python3 balanced_auto_exception.py hbase code ${test_rate} &> CV/real_final/${test_rate}/hbase-code.txt
python3 balanced_auto_exception.py guava code ${test_rate} &> CV/real_final/${test_rate}/guava-code.txt
python3 balanced_auto_exception.py glassfish code ${test_rate} &> CV/real_final/${test_rate}/glassfish-code.txt


python3 merge_network_2_wordvec_model.py guava ${test_rate} &> CV/real_final/${test_rate}/guava-CAST_m.txt
python3 merge_network_2_wordvec_model.py hbase ${test_rate} &> CV/real_final/${test_rate}/hbase-CAST_m.txt
python3 merge_network_2_wordvec_model.py glassfish ${test_rate} &> CV/real_final/${test_rate}/glassfish-CAST_m.txt

python3 doc_product.py guava ${test_rate} &> CV/real_final/${test_rate}/guava-CAST_d.txt
python3 doc_product.py hbase ${test_rate} &> CV/real_final/${test_rate}/hbase-CAST_d.txt
python3 doc_product.py glassfish ${test_rate} &> CV/real_final/${test_rate}/glassfish-CAST_d.txt

:<<'END'
END
