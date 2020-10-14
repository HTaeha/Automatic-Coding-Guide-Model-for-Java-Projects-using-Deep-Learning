#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "LoggingTree.h"

#define MAX_NUM 1000

int digit_check(int num);
int depth_check(char* bup, int count_tab);
int check(int depth, int count_tab, char* bup, char* str, int digit);
void printTree(Tree *T, int isThrow, char *str_path, char *str_method, FILE *fp_w, int last);
void printTree_preorder(Tree *T, int isThrow, char *str_path, char *str_method, FILE *fp_w, int last);

int isThrow = 0;
int isTrycatch = 0;
int throw_depth = MAX_NUM;
char *str_throw = "THROW";

int main(){
	
	FILE *fp_r;
	FILE *fp_w;
	char bup[1000];
	char str[1000];
	char str_m[100];
	char str_p[1000];
	int ret;
	int i;
	int numOfsnippets = 0;
	int depth=0;
	TreeNode *parentNode;
	int count_tab = 0;
	int index_str = 0;
	int index_str_p = 0;
    int path_flag = 0; //It is flag to start building tree.
    int valid = 0; //It is flag to valid node.
    int code_snippet_depth = -1;
    int digit = 0;

    int tree_based = 0; //0 : preorder
                        //1 : tree_based

//	fp_r = fopen("input/Sample/Sample-real.txt","r");
//	fp_w = fopen("output/Sample/Sample-real.json", "w");
	fp_r = fopen("input/Throw/hbase-CAST_s.txt","r");
	fp_w = fopen("output/Throw/hbase-CAST_s.json", "w");


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
        while(bup[count_tab] == '\t'){
			count_tab++;
		}
        if(bup[strlen(bup)-1] == '\n'){
            bup[strlen(bup)-2] = 0;
        }else{
            bup[strlen(bup)] = 0;
        }

		str[0] = 0;

        if(strncmp(bup, "method:", 7) == 0){
			str_m[0] = 0;
			strcat(str_m, bup+8);
            throw_depth = MAX_NUM;
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
            path_flag = 1;
            valid = 1;
            isTrycatch = 0;
            code_snippet_depth = -1;
            continue;
        }

        depth = depth_check(bup, count_tab);
        digit = digit_check(depth);
        if(depth <= code_snippet_depth){
            valid = 0;
        }

        if(valid){
            if(bup[count_tab+digit-1] <= '9' && bup[count_tab+digit-1] >= '0'){
                if(!check(depth, count_tab, bup, str, digit)){
                    continue;
                }
            }else{
                continue;
            }/*
            if(bup[count_tab] <= '9' && bup[count_tab] >= '0'){
                if(bup[1+count_tab] <= '9' && bup[1+count_tab] >= '0'){
                    if(bup[2+count_tab] <= '9' && bup[2+count_tab] >= '0'){
                        if(!check(depth, count_tab, bup, str, 3)){
                            continue;
                        }
                    }else{
                        if(!check(depth, count_tab, bup, str, 2)){
                            continue;
                        }
                    }
                }else{
                    if(!check(depth, count_tab, bup, str, 1)){
                        continue;
                    }
                }
            }else{
                continue;
            } */
        }else{
            continue;
        }

        if(depth <= throw_depth){
            throw_depth = MAX_NUM;
        }
        

        /*  if child num is over 1024, don't insert another node in tree.
        if(check_child_num(T->root) > 0){
            printf("%d\n", check_child_num(T->root));
            continue;
        }*/

        //After 'path:' insert root node.
        if(path_flag == 1){
            code_snippet_depth = depth;
            insert(T, str, NULL, depth);
            path_flag = 0;
        }else{ //Insert node.
            if(numOfsnippets != 0 && throw_depth >= depth){
                parentNode = searchParentNode(T->root, depth-1);	
                insert(T, str, parentNode, depth);
            }
        }
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
                return (bup[count_tab]-'0')*10 + (bup[1+count_tab] - '0');
            }
        }else{
            return bup[count_tab] - '0';
        }
    }
}
int check(int depth, int count_tab, char* bup, char* str, int digit){
    int index_str = 0;
    int annotation_start = 0;
    int i;

    if(strncmp(bup+digit+count_tab, str_throw,5) == 0){
        isThrow = 1;
        return 0;
    }else{
        if(strncmp(bup+digit+count_tab, "TryStmt", 7) == 0|| strncmp(bup+digit+count_tab,"CatchClause" , 11) == 0){
            isTrycatch = 1;
        }else if(strncmp(bup+digit+count_tab, "ThrowStmt", 9) == 0){
            throw_depth = depth;
            return 0;
        }
        for(i=digit+count_tab;i<strlen(bup);i++){
            if(bup[i] == '\\'){
                continue;
            }else if(bup[i] == '/' && bup[i+1] == '*'){
                annotation_start = 1;
                continue;
            }else if(bup[i] == '*' && bup[i+1] == '/'){
                annotation_start = 0;
                i++;
                continue;
            }

            if(annotation_start == 1){
                continue;
            }else{
                str[index_str++] = bup[i];
            }

        }
        str[index_str] = 0;
    }
    return 1;
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
void printTree_preorder(Tree *T, int isThrow, char *str_path, char *str_method, FILE *fp_w, int last){
    TreeNode * node;
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
