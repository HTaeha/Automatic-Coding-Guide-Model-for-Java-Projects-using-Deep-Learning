#!/bin/bash
filename=guava

python3 balanced_auto_logging.py ${filename} bytecode_logging 1 &> CV/1/${filename}-bytecode_logging.txt
python3 balanced_auto_logging.py ${filename} bytecode_logging 2 &> CV/2/${filename}-bytecode_logging.txt
python3 balanced_auto_logging.py ${filename} bytecode_logging 3 &> CV/3/${filename}-bytecode_logging.txt
