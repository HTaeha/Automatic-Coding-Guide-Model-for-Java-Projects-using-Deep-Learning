#!/bin/bash
test_rate=1
min_word_frequency=5
LSTM=64
:<<END
END
python3 DecisionTree_Logging.py hbase code ${test_rate} ${min_word_frequency} ${LSTM}&> CV/Logging/${test_rate}/hbase-DecisionTree-code_frequency${min_word_frequency}_LSTM${LSTM}_batch64.txt
python3 DecisionTree_Logging.py hbase AST ${test_rate} ${min_word_frequency} ${LSTM}&> CV/Logging/${test_rate}/hbase-DecisionTree-AST_frequency${min_word_frequency}_LSTM${LSTM}_batch64.txt
python3 DecisionTree_Logging.py hbase CAST ${test_rate} ${min_word_frequency} ${LSTM}&> CV/Logging/${test_rate}/hbase-DecisionTree-CAST_frequency${min_word_frequency}_LSTM${LSTM}_batch64.txt

min_word_frequency=500
LSTM=64
python3 DecisionTree_Logging.py hbase code ${test_rate} ${min_word_frequency} ${LSTM}&> CV/Logging/${test_rate}/hbase-DecisionTree-code_frequency${min_word_frequency}_LSTM${LSTM}_batch64.txt
python3 DecisionTree_Logging.py hbase AST ${test_rate} ${min_word_frequency} ${LSTM}&> CV/Logging/${test_rate}/hbase-DecisionTree-AST_frequency${min_word_frequency}_LSTM${LSTM}_batch64.txt
python3 DecisionTree_Logging.py hbase CAST ${test_rate} ${min_word_frequency} ${LSTM}&> CV/Logging/${test_rate}/hbase-DecisionTree-CAST_frequency${min_word_frequency}_LSTM${LSTM}_batch64.txt
min_word_frequency=5000
LSTM=64
python3 DecisionTree_Logging.py hbase code ${test_rate} ${min_word_frequency} ${LSTM}&> CV/Logging/${test_rate}/hbase-DecisionTree-code_frequency${min_word_frequency}_LSTM${LSTM}_batch64.txt
python3 DecisionTree_Logging.py hbase AST ${test_rate} ${min_word_frequency} ${LSTM}&> CV/Logging/${test_rate}/hbase-DecisionTree-AST_frequency${min_word_frequency}_LSTM${LSTM}_batch64.txt
python3 DecisionTree_Logging.py hbase CAST ${test_rate} ${min_word_frequency} ${LSTM}&> CV/Logging/${test_rate}/hbase-DecisionTree-CAST_frequency${min_word_frequency}_LSTM${LSTM}_batch64.txt
min_word_frequency=10000
LSTM=64
python3 DecisionTree_Logging.py hbase code ${test_rate} ${min_word_frequency} ${LSTM}&> CV/Logging/${test_rate}/hbase-DecisionTree-code_frequency${min_word_frequency}_LSTM${LSTM}_batch64.txt
python3 DecisionTree_Logging.py hbase AST ${test_rate} ${min_word_frequency} ${LSTM}&> CV/Logging/${test_rate}/hbase-DecisionTree-AST_frequency${min_word_frequency}_LSTM${LSTM}_batch64.txt
python3 DecisionTree_Logging.py hbase CAST ${test_rate} ${min_word_frequency} ${LSTM}&> CV/Logging/${test_rate}/hbase-DecisionTree-CAST_frequency${min_word_frequency}_LSTM${LSTM}_batch64.txt


