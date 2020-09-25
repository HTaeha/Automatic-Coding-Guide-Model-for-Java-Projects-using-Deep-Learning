#!/bin/bash

python3 analysis_True.py hbase code AST CAST CAST_s 1 &> test_pred.txt #Analysis/Orthogonality_check/True/hbase1.txt
:<<END
python3 analysis_True.py hbase code AST CAST CAST_s 2 &> Analysis/Orthogonality_check/True/hbase2.txt
python3 analysis_True.py hbase code AST CAST CAST_s 3 &> Analysis/Orthogonality_check/True/hbase3.txt
python3 analysis_True.py hbase code AST CAST CAST_s 4 &> Analysis/Orthogonality_check/True/hbase4.txt
python3 analysis_True.py hbase code AST CAST CAST_s 5 &> Analysis/Orthogonality_check/True/hbase5.txt

python3 analysis_True.py glassfish code AST CAST CAST_s 1 &> Analysis/Orthogonality_check/True/glassfish1.txt
python3 analysis_True.py glassfish code AST CAST CAST_s 2 &> Analysis/Orthogonality_check/True/glassfish2.txt
python3 analysis_True.py glassfish code AST CAST CAST_s 3 &> Analysis/Orthogonality_check/True/glassfish3.txt
python3 analysis_True.py glassfish code AST CAST CAST_s 4 &> Analysis/Orthogonality_check/True/glassfish4.txt
python3 analysis_True.py glassfish code AST CAST CAST_s 5 &> Analysis/Orthogonality_check/True/glassfish5.txt

python3 analysis_True.py guava code AST CAST CAST_s 1 &> Analysis/Orthogonality_check/True/guava1.txt
python3 analysis_True.py guava code AST CAST CAST_s 2 &> Analysis/Orthogonality_check/True/guava2.txt
python3 analysis_True.py guava code AST CAST CAST_s 3 &> Analysis/Orthogonality_check/True/guava3.txt
python3 analysis_True.py guava code AST CAST CAST_s 4 &> Analysis/Orthogonality_check/True/guava4.txt
python3 analysis_True.py guava code AST CAST CAST_s 5 &> Analysis/Orthogonality_check/True/guava5.txt
END
