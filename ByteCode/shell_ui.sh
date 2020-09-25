#!/bin/bash
filename=guava

python3 balanced_auto_logging.py ${filename} bytecode_logging 4 &> CV/4/${filename}-bytecode_logging.txt
python3 balanced_auto_logging.py ${filename} bytecode_logging 5 &> CV/5/${filename}-bytecode_logging.txt
python3 balanced_auto_logging.py ${filename} bytecode_logging 6 &> CV/6/${filename}-bytecode_logging.txt
