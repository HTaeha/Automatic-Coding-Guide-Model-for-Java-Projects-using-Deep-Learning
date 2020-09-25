#!/bin/bash
input_data="hbase"
input_type="code"

python3 analysis_fileIO.py ${input_data} ${input_type} 500 &> Analysis/vocabulary/${input_data}-${input_type}_frequency500.txt
#python3 analysis_fileIO.py ${input_data} ${input_type} 1000 &> Analysis/vocabulary/${input_data}-${input_type}_no_punc_frequency1000.txt
python3 analysis_fileIO.py ${input_data} ${input_type} 5000 &> Analysis/vocabulary/${input_data}-${input_type}_frequency5000.txt
python3 analysis_fileIO.py ${input_data} ${input_type} 10000 &> Analysis/vocabulary/${input_data}-${input_type}_frequency10000.txt
