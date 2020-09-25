#!/bin/bash
:<<END
END
python3 analysis_False.py hbase code AST CAST CAST_s 1 &> Analysis/Orthogonality_check/False/hbase1.txt
python3 analysis_False.py hbase code AST CAST CAST_s 2 &> Analysis/Orthogonality_check/False/hbase2.txt
python3 analysis_False.py hbase code AST CAST CAST_s 3 &> Analysis/Orthogonality_check/False/hbase3.txt
python3 analysis_False.py hbase code AST CAST CAST_s 4 &> Analysis/Orthogonality_check/False/hbase4.txt
python3 analysis_False.py hbase code AST CAST CAST_s 5 &> Analysis/Orthogonality_check/False/hbase5.txt
:<<END
END
python3 analysis_False.py glassfish code AST CAST CAST_s 1 &> Analysis/Orthogonality_check/False/glassfish1.txt
python3 analysis_False.py glassfish code AST CAST CAST_s 2 &> Analysis/Orthogonality_check/False/glassfish2.txt
python3 analysis_False.py glassfish code AST CAST CAST_s 3 &> Analysis/Orthogonality_check/False/glassfish3.txt
python3 analysis_False.py glassfish code AST CAST CAST_s 4 &> Analysis/Orthogonality_check/False/glassfish4.txt
python3 analysis_False.py glassfish code AST CAST CAST_s 5 &> Analysis/Orthogonality_check/False/glassfish5.txt
:<<END
END
python3 analysis_False.py guava code AST CAST CAST_s 1 &> Analysis/Orthogonality_check/False/guava1.txt
python3 analysis_False.py guava code AST CAST CAST_s 2 &> Analysis/Orthogonality_check/False/guava2.txt
python3 analysis_False.py guava code AST CAST CAST_s 3 &> Analysis/Orthogonality_check/False/guava3.txt
python3 analysis_False.py guava code AST CAST CAST_s 4 &> Analysis/Orthogonality_check/False/guava4.txt
python3 analysis_False.py guava code AST CAST CAST_s 5 &> Analysis/Orthogonality_check/False/guava5.txt
