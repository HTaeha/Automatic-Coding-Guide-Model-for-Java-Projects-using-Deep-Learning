#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "LoggingTree.h"

int digit_check(int num);
int depth_check(char* bup, int count_tab);
void printTree_preorder(Tree *T, int isThrow, char *str_path, char * str_method, FILE * fp_w, int last);
void printTree(Tree *T, int isThrow, char *str_path, char *str_method, FILE *fp_w, int last);
int main(){
	
	FILE *fp_r;
	FILE *fp_w;
	char bup[1000];
	char str[1000];
	char str_m[100];
	char str_p[1000];
	int ret;
	int isThrow = 0;
    int isTrycatch = 0;
	int i;
    char *str_trycatch = "TRY-CATCH";
    char *str_throw = "THROW";
	int numOfsnippets = 0;
	int depth;
    int temp_depth;
    int code_snippet_depth = -1;
    int trycatch_depth;
    int valid=0;
    int digit;
	TreeNode *parentNode;
	int count_tab = 0;
	int index_str = 0;
	int index_str_p = 0;
	int annotation_start = 0;

    int tree_based = 0; //0 : preorder
                        //1 : tree_based

//	fp_r = fopen("input/Sample/Sample-real.txt","r");
//	fp_w = fopen("output/Sample/Sample-real.json", "w");
	fp_r = fopen("input/Throw/guava-code.txt","r");
	fp_w = fopen("output/Throw/guava-code.json", "w");


	if(fp_r == NULL || fp_w == NULL){
		puts("File open failed!");
		return -1;
	}

	Tree *T = NodeCreate();

	fprintf(fp_w, "[\r\n");
	while(1){
		if(fgets(bup, sizeof(bup), fp_r) == NULL){
			break;
		}
        count_tab = 0;
        while(bup[count_tab] == '\t' || bup[count_tab] == ' '){
            count_tab++;
        }
        if(bup[strlen(bup)-1] == '\n'){
            bup[strlen(bup)-2] = 0;
        }else{
            bup[strlen(bup)] = 0;
        }

        str[0] = 0;

        temp_depth = depth_check(bup, count_tab);
        digit = digit_check(temp_depth);
        if(temp_depth != -1 && bup[count_tab+digit]>='A' && bup[count_tab+digit]<='Z'){
            if(strncmp(bup+count_tab+digit, "MethodDeclaration", 17) == 0){
                code_snippet_depth = temp_depth;
                valid = 1;
                continue;
            }else if(strncmp(bup+count_tab+digit, "TryStmt", 7) == 0 || strncmp(bup+count_tab+digit, "CatchClause", 11) == 0 || strncmp(bup+count_tab+digit, str_throw, 5) == 0){
                if(temp_depth <= code_snippet_depth || !valid){
                    valid = 0;
                    continue;
                }
            }else{
                if(temp_depth <= code_snippet_depth){
                    valid = 0;
                }
                continue;
            }
        }

        if(strncmp(bup, "method:", 7) == 0){
            str_m[0] = 0;
            strcat(str_m, bup+8);
            continue;
        }else if(strncmp(bup, "path:", 5) == 0){
            if(numOfsnippets != 0 && isTrycatch){
                if(tree_based){
                    printTree(T, isThrow, str_p, str_m, fp_w, 0);
                }else{
                    printTree_preorder(T, isThrow, str_p, str_m, fp_w, 0);
                }
                ClearTree(T->root);
                free(T);
                isThrow = 0;

                Tree *T = NodeCreate();
            }
            str_p[0] = 0;
            for(i = 6;i<strlen(bup);i++){
                if(bup[i] == '/'){
                    str_p[index_str_p++] = '/';
                    str_p[index_str_p++] = '/';
                }else{
                    str_p[index_str_p++] = bup[i];
                }
            }
            str_p[index_str_p] = 0;
            index_str_p = 0;
            numOfsnippets++;
            isTrycatch = 0;
            depth = 0;
            valid = 1;
            continue;
        }else{
            if(strncmp(bup+digit+count_tab, str_throw, 5) == 0 && valid){
                isThrow = 1;
                continue;
            }else if((strncmp(bup+digit+count_tab, "TryStmt", 7) == 0 || strncmp(bup+count_tab+digit, "CatchClause", 11) == 0)&& valid){
                isTrycatch = 1;
                continue;
            }else if(strncmp(bup+count_tab, "Optional[", 9) == 0){
                strcat(str, bup+9);
            }else if(strncmp(bup+count_tab, "Optional.empty", 14) == 0){
                continue;
            }else if(strncmp(bup+count_tab, "}]", 2) == 0){
                str[0] = '}';
                str[1] = 0;
            }else if(strncmp(bup+count_tab, "//", 2) == 0){
                continue;
            }else if(strncmp(bup+count_tab, "/*", 2) == 0){
                if(strncmp(bup+strlen(bup)-2, "*/", 2) == 0){
                    continue;
                }else{
                    annotation_start = 1;
                    continue;
                }
            }else if(strncmp(bup+strlen(bup)-2, "*/", 2) == 0){
                annotation_start = 0;
                continue;
            }else{
                if(annotation_start == 0){
                    for(i=count_tab;i<strlen(bup);i++){
                        if(bup[i] == '\\' || bup[i] == '\"' || bup[i] == '\r'){
                            continue;
                        }else{
                            str[index_str++] = bup[i];
                        }
                    }
                    str[index_str] = 0;
                    index_str = 0;
                }else{
                    continue;
                }
            }
        } 


        parentNode = searchParentNode(T->root, depth-1);	
        insert(T, str, parentNode, depth);
		depth++;
	}
    //	print_preorder(T->root);
    if(isTrycatch){
        if(tree_based){
            printTree(T, isThrow, str_p, str_m, fp_w, 1);
        }else{
            printTree_preorder(T, isThrow, str_p, str_m, fp_w, 1);
        }
    }

	ClearTree(T->root);
	free(T);

	fprintf(fp_w, "]");

	fclose(fp_r);
	fclose(fp_w);

	return 0;
}
int digit_check(int num){
    int result = 0;
    while(num > 0){
        num /= 10;
        result++;
    }
    return result;
}
int depth_check(char* bup, int count_tab){
    if(bup[count_tab] <= '9' && bup[count_tab] >= '0'){
        if(bup[1+count_tab] <= '9' && bup[1+count_tab] >= '0'){
            if(bup[2+count_tab] <= '9' && bup[2+count_tab] >= '0'){
                return (bup[count_tab]-'0')*100 + (bup[1+count_tab] - '0')*10 + (bup[2+count_tab] - '0');
            }else{
                return (bup[count_tab] - '0')*10 + (bup[1+count_tab] - '0');
            }
        }else{
            return bup[count_tab] - '0';
        }
    }else{
        return -1;
    }
}
void printTree(Tree *T, int isThrow, char *str_path, char *str_method, FILE *fp_w, int last){
	TreeNode *node;
	Queue q;
	InitQueue(&q);
	Stack s;
	InitStack(&s);
	TreeNode *temp;

	searchLeafNode(T->root, &q);

	fprintf(fp_w, "\t{\r\n");
	fprintf(fp_w, "\t\t\"sentence\" : \"");
	while(!IsEmpty(&q)){
		node = Peek(&q);
		DeQueue(&q);

		while(1){
			Push(&s, node);
			if(node == T->root){
				break;
			}
			node = node->parent;
		}
		while(!IsEmptyS(&s)){
			temp = Pop(&s);
			fprintf(fp_w, "%s ", temp->struc);
		}
	}
	fprintf(fp_w, "\",\r\n");
	fprintf(fp_w, "\t\t\"isThrow\" : %d,\r\n", isThrow);
	fprintf(fp_w, "\t\t\"path\" : \"%s\",\r\n", str_path);
	fprintf(fp_w, "\t\t\"method\" : \"%s\"\r\n", str_method);
	if(last == 0){
		fprintf(fp_w, "\t},\r\n");
	}else{
		fprintf(fp_w, "\t}\r\n");
	}
}
void printTree_preorder(Tree *T, int isThrow, char *str_path, char * str_method, FILE * fp_w, int last){
	TreeNode *node;
	int i;

	fprintf(fp_w, "\t{\r\n");
	fprintf(fp_w, "\t\t\"sentence\" : \"");

	Stack s;
	InitStack(&s);

	node = T->root;
	Push(&s, node);

	while(IsEmptyS(&s) == false){
		node = Pop(&s);
		fprintf(fp_w, "%s ", node->struc);

		for(i=node->child_idx-1;i>=0;i--){
			Push(&s, node->child[i]);
		}
	}

	fprintf(fp_w, "\",\r\n");
	fprintf(fp_w, "\t\t\"isThrow\" : %d,\r\n", isThrow);
	fprintf(fp_w, "\t\t\"path\" : \"%s\",\r\n", str_path);
	fprintf(fp_w, "\t\t\"method\" : \"%s\"\r\n", str_method);
	if(last == 0){
		fprintf(fp_w, "\t},\r\n");
	}else{
		fprintf(fp_w, "\t}\r\n");
	}
}
