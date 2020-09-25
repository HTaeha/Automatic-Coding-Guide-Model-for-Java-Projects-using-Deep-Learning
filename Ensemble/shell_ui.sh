#!/bin/bash
test_rate=1

python3 Ensemble_Bagging_Exception.py hbase ${test_rate} &> CV/Exception/${test_rate}/hbase-ensemble_code,AST.txt
python3 Ensemble_Bagging_Exception.py glassfish ${test_rate} &> CV/Exception/${test_rate}/glassfish-ensemble_code,AST.txt
python3 Ensemble_Bagging_Exception.py guava ${test_rate} &> CV/Exception/${test_rate}/guava-ensemble_code,AST.txt
