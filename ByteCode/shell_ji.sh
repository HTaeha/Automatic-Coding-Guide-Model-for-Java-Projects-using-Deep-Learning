#!/bin/bash
filename=guava

python3 balanced_auto_logging.py ${filename} bytecode_logging 9 &> CV/9/${filename}-bytecode_logging.txt
python3 balanced_auto_logging.py ${filename} bytecode_logging 10 &> CV/10/${filename}-bytecode_logging.txt
