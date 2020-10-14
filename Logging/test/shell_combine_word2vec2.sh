#!/bin/bash
min_word_frequency=5

python3 combine_all_project.py hbase guava glassfish code ${min_word_frequency}
python3 combine_all_project.py hbase guava glassfish AST ${min_word_frequency}
python3 combine_all_project.py hbase guava glassfish CAST ${min_word_frequency}
python3 combine_all_project.py hbase guava glassfish CAST_s ${min_word_frequency}
min_word_frequency=500

python3 combine_all_project.py hbase guava glassfish code ${min_word_frequency}
python3 combine_all_project.py hbase guava glassfish AST ${min_word_frequency}
python3 combine_all_project.py hbase guava glassfish CAST ${min_word_frequency}
python3 combine_all_project.py hbase guava glassfish CAST_s ${min_word_frequency}
min_word_frequency=5000

python3 combine_all_project.py hbase guava glassfish code ${min_word_frequency}
python3 combine_all_project.py hbase guava glassfish AST ${min_word_frequency}
python3 combine_all_project.py hbase guava glassfish CAST ${min_word_frequency}
python3 combine_all_project.py hbase guava glassfish CAST_s ${min_word_frequency}
min_word_frequency=10000

python3 combine_all_project.py hbase guava glassfish code ${min_word_frequency}
python3 combine_all_project.py hbase guava glassfish AST ${min_word_frequency}
python3 combine_all_project.py hbase guava glassfish CAST ${min_word_frequency}
python3 combine_all_project.py hbase guava glassfish CAST_s ${min_word_frequency}

