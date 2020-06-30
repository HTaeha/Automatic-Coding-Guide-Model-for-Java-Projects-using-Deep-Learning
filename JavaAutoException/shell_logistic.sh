#!/bin/bash

test_rate=2
python3 logistic_regression.py hbase code ${test_rate} 5 64 &> logistic${test_rate}.txt
test_rate=3
python3 logistic_regression.py hbase code ${test_rate} 5 64 &> logistic${test_rate}.txt
test_rate=4
python3 logistic_regression.py hbase code ${test_rate} 5 64 &> logistic${test_rate}.txt
test_rate=5
python3 logistic_regression.py hbase code ${test_rate} 5 64 &> logistic${test_rate}.txt
test_rate=6
python3 logistic_regression.py hbase code ${test_rate} 5 64 &> logistic${test_rate}.txt
test_rate=7
python3 logistic_regression.py hbase code ${test_rate} 5 64 &> logistic${test_rate}.txt
test_rate=8
python3 logistic_regression.py hbase code ${test_rate} 5 64 &> logistic${test_rate}.txt
test_rate=9
python3 logistic_regression.py hbase code ${test_rate} 5 64 &> logistic${test_rate}.txt
test_rate=10
python3 logistic_regression.py hbase code ${test_rate} 5 64 &> logistic${test_rate}.txt

