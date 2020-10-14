#!/bin/bash

:<<END
END
python3 analysis_False.py hbase code AST CAST CAST_s 1 &> Analysis/Orthogonality_check/False/real_final/hbase1.txt
python3 analysis_False.py hbase code AST CAST CAST_s 2 &> Analysis/Orthogonality_check/False/real_final/hbase2.txt
python3 analysis_False.py hbase code AST CAST CAST_s 3 &> Analysis/Orthogonality_check/False/real_final/hbase3.txt
python3 analysis_False.py hbase code AST CAST CAST_s 4 &> Analysis/Orthogonality_check/False/real_final/hbase4.txt
python3 analysis_False.py hbase code AST CAST CAST_s 5 &> Analysis/Orthogonality_check/False/real_final/hbase5.txt

python3 analysis_False.py glassfish code AST CAST CAST_s 1 &> Analysis/Orthogonality_check/False/real_final/glassfish1.txt
python3 analysis_False.py glassfish code AST CAST CAST_s 2 &> Analysis/Orthogonality_check/False/real_final/glassfish2.txt
python3 analysis_False.py glassfish code AST CAST CAST_s 3 &> Analysis/Orthogonality_check/False/real_final/glassfish3.txt
python3 analysis_False.py glassfish code AST CAST CAST_s 4 &> Analysis/Orthogonality_check/False/real_final/glassfish4.txt
python3 analysis_False.py glassfish code AST CAST CAST_s 5 &> Analysis/Orthogonality_check/False/real_final/glassfish5.txt

python3 analysis_False.py guava code AST CAST CAST_s 1 &> Analysis/Orthogonality_check/False/real_final/guava1.txt
python3 analysis_False.py guava code AST CAST CAST_s 2 &> Analysis/Orthogonality_check/False/real_final/guava2.txt
python3 analysis_False.py guava code AST CAST CAST_s 3 &> Analysis/Orthogonality_check/False/real_final/guava3.txt
python3 analysis_False.py guava code AST CAST CAST_s 4 &> Analysis/Orthogonality_check/False/real_final/guava4.txt
python3 analysis_False.py guava code AST CAST CAST_s 5 &> Analysis/Orthogonality_check/False/real_final/guava5.txt
