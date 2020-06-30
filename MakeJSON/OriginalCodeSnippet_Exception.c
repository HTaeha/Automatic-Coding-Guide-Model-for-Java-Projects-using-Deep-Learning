#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "LoggingTree.h"

#define MAX_BUP_LEN 10000

int exception_check(char * str1, char * str2);
int digit_check(int num);
int depth_check(char* bup, int count_tab);
void printTree_preorder(Tree *T, int isException, char *str_path, char * str_method, FILE * fp_w, int last);
void printTree(Tree *T, int isException, char *str_path, char *str_method, FILE *fp_w, int last);

char *str_exception = "EXCEPTION";
char *str_try = "TryStmt";
char *str_catch = "CatchClause";
char *str_throw = "ThrowStmt";
char *str_exception_method = "-Exception_method";

int main(){

	FILE *fp_r;
	FILE *fp_w;
	char bup[MAX_BUP_LEN];
	char str[MAX_BUP_LEN];
	char str_m[100];
	char str_p[1000];
	int ret;
	int isException = 0;
	int i;
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
	fp_r = fopen("input/Exception/real_last/glassfish-code.txt","r");
	fp_w = fopen("output/Exception/real_last/glassfish-code.json", "w");


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
            if(strncmp(bup+count_tab+digit, "MethodDeclaration", 17) == 0 && temp_depth == 2){
                code_snippet_depth = temp_depth;
                valid = 1;
                continue;
            //}else if(strncmp(bup+count_tab+digit, str_exception, 9) == 0){
            }else if(exception_check(bup+count_tab+digit, bup+strlen(bup)-17)){
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
			if(numOfsnippets != 0){
                if(tree_based){
                    printTree(T, isException, str_p, str_m, fp_w, 0);
                }else{
                    printTree_preorder(T, isException, str_p, str_m, fp_w, 0);
                }
				ClearTree(T->root);
				free(T);
				isException = 0;

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
			depth = 0;
			valid = 1;
			continue;
		}else{
			//if(strncmp(bup+digit+count_tab, str_exception, 9) == 0 && valid){
			if(exception_check(bup+digit+count_tab, bup+strlen(bup)-17) && valid){
				isException = 1;
				continue;
			}else if(strncmp(bup+count_tab, "Optional[", 9) == 0){
				strcat(str, bup+9);
			}else if(strncmp(bup+count_tab, "Optional.empty", 14) == 0){
				continue;
			}else if(strncmp(bup+count_tab, "}]", 2) == 0){
				str[0] = '}';
				str[1] = 0;
			}else{
                for(i=count_tab;i<strlen(bup);i++){
                    if(bup[i] == '\\' || bup[i] == '\"' || bup[i] == '\r'){
                        continue;
                    }else{
                        str[index_str++] = bup[i];
                    }
                }
                str[index_str] = 0;
                index_str = 0;
            }
        } 

        parentNode = searchParentNode(T->root, depth-1);	
		insert(T, str, parentNode, depth);
		depth++;
	}
    //	print_preorder(T->root);
    if(tree_based){
        printTree(T, isException, str_p, str_m, fp_w, 1);
    }else{
        printTree_preorder(T, isException, str_p, str_m, fp_w, 1);
    }

	ClearTree(T->root);
	free(T);

	fprintf(fp_w, "]");

	fclose(fp_r);
	fclose(fp_w);

	return 0;
}
int exception_check(char * str1, char * str2){
    if(strncmp(str1, str_exception,9) == 0 || strncmp(str1, str_try, 7) == 0|| strncmp(str1, str_catch, 11) == 0|| strncmp(str1, str_throw, 9) == 0 || strncmp(str2, str_exception_method, 17) == 0){
        return 1;
    }else{
        return 0;
    }
}
int digit_check(int num){
    int result = 0;
    if(num == 0){
        return 1;
    }
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
void printTree(Tree *T, int isException, char *str_path, char *str_method, FILE *fp_w, int last){
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
	fprintf(fp_w, "\t\t\"isException\" : %d,\r\n", isException);
	fprintf(fp_w, "\t\t\"path\" : \"%s\",\r\n", str_path);
	fprintf(fp_w, "\t\t\"method\" : \"%s\"\r\n", str_method);
	if(last == 0){
		fprintf(fp_w, "\t},\r\n");
	}else{
		fprintf(fp_w, "\t}\r\n");
	}
}
void printTree_preorder(Tree *T, int isException, char *str_path, char * str_method, FILE * fp_w, int last){
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
	fprintf(fp_w, "\t\t\"isException\" : %d,\r\n", isException);
	fprintf(fp_w, "\t\t\"path\" : \"%s\",\r\n", str_path);
	fprintf(fp_w, "\t\t\"method\" : \"%s\"\r\n", str_method);
	if(last == 0){
		fprintf(fp_w, "\t},\r\n");
	}else{
		fprintf(fp_w, "\t}\r\n");
	}
}
