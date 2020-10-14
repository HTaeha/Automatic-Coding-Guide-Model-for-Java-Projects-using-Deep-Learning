#!/bin/bash
filename=guava

python3 balanced_auto_logging.py ${filename} bytecode_logging 7 &> CV/7/${filename}-bytecode_logging.txt
python3 balanced_auto_logging.py ${filename} bytecode_logging 8 &> CV/8/${filename}-bytecode_logging.txt
