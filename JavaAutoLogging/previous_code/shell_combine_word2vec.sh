#!/bin/bash
min_word_frequency=5

python3 combine_all_project.py glassfish guava hbase code ${min_word_frequency}
python3 combine_all_project.py glassfish guava hbase AST ${min_word_frequency}
python3 combine_all_project.py glassfish guava hbase CAST ${min_word_frequency}
python3 combine_all_project.py glassfish guava hbase CAST_s ${min_word_frequency}
min_word_frequency=500

python3 combine_all_project.py glassfish guava hbase code ${min_word_frequency}
python3 combine_all_project.py glassfish guava hbase AST ${min_word_frequency}
python3 combine_all_project.py glassfish guava hbase CAST ${min_word_frequency}
python3 combine_all_project.py glassfish guava hbase CAST_s ${min_word_frequency}
min_word_frequency=5000

python3 combine_all_project.py glassfish guava hbase code ${min_word_frequency}
python3 combine_all_project.py glassfish guava hbase AST ${min_word_frequency}
python3 combine_all_project.py glassfish guava hbase CAST ${min_word_frequency}
python3 combine_all_project.py glassfish guava hbase CAST_s ${min_word_frequency}
min_word_frequency=10000

python3 combine_all_project.py glassfish guava hbase code ${min_word_frequency}
python3 combine_all_project.py glassfish guava hbase AST ${min_word_frequency}
python3 combine_all_project.py glassfish guava hbase CAST ${min_word_frequency}
python3 combine_all_project.py glassfish guava hbase CAST_s ${min_word_frequency}

