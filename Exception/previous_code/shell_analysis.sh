#!/bin/bash
input_data="glassfish"
input_type="code"
limit_count=4

python3 analysis_fileIO.py ${input_data} ${input_type} 2 ${limit_count} &> Analysis/keyword/${input_data}-${input_type}_no_punc_2:1_limit${limit_count}.txt
python3 analysis_fileIO.py ${input_data} ${input_type} 3 ${limit_count} &> Analysis/keyword/${input_data}-${input_type}_no_punc_3:1_limit${limit_count}.txt
python3 analysis_fileIO.py ${input_data} ${input_type} 4 ${limit_count} &> Analysis/keyword/${input_data}-${input_type}_no_punc_4:1_limit${limit_count}.txt
python3 analysis_fileIO.py ${input_data} ${input_type} 5 ${limit_count} &> Analysis/keyword/${input_data}-${input_type}_no_punc_5:1_limit${limit_count}.txt
python3 analysis_fileIO.py ${input_data} ${input_type} 6 ${limit_count} &> Analysis/keyword/${input_data}-${input_type}_no_punc_6:1_limit${limit_count}.txt
