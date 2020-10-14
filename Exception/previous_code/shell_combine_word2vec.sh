#!/bin/bash
min_word_frequency=500

python3 combine_all_project.py hbase glassfish guava code ${min_word_frequency}
python3 combine_all_project.py hbase glassfish guava AST ${min_word_frequency}
python3 combine_all_project.py hbase glassfish guava CAST ${min_word_frequency}
python3 combine_all_project.py hbase glassfish guava CAST_s ${min_word_frequency}
python3 combine_all_project_code,AST.py hbase glassfish guava ${min_word_frequency}
min_word_frequency=5000

python3 combine_all_project.py hbase glassfish guava code ${min_word_frequency}
python3 combine_all_project.py hbase glassfish guava AST ${min_word_frequency}
python3 combine_all_project.py hbase glassfish guava CAST ${min_word_frequency}
python3 combine_all_project.py hbase glassfish guava CAST_s ${min_word_frequency}
python3 combine_all_project_code,AST.py hbase glassfish guava ${min_word_frequency}
min_word_frequency=10000

python3 combine_all_project.py hbase glassfish guava code ${min_word_frequency}
python3 combine_all_project.py hbase glassfish guava AST ${min_word_frequency}
python3 combine_all_project.py hbase glassfish guava CAST ${min_word_frequency}
python3 combine_all_project.py hbase glassfish guava CAST_s ${min_word_frequency}
python3 combine_all_project_code,AST.py hbase glassfish guava ${min_word_frequency}

