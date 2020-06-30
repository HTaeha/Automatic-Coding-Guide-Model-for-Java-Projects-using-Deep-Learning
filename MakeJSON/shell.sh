#!/bin/bash

#gcc AST_Logging_BuildTreeAndPrint.c LoggingTree.c
#gcc AST_Exception_BuildTreeAndPrint.c LoggingTree.c
#gcc OriginalCodeSnippet_Logging.c LoggingTree.c
#gcc OriginalCodeSnippet_Exception.c LoggingTree.c
gcc Num_AST_Logging_BuildTreeAndPrint.c LoggingTree.c
#gcc Num_AST_Exception_BuildTreeAndPrint.c LoggingTree.c
#gcc Add_{}_AST_Logging_BuildTreeAndPrint.c LoggingTree.c
./a.out
