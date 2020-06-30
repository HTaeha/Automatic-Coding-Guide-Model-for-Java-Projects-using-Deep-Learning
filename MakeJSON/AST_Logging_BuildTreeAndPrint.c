#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "LoggingTree.h"

#define MAX_BUP_LEN 10000

int digit_check(int num);
int depth_check(char* bup, int count_tab);
int check(int depth, int count_tab, char* bup, char* str, int digit);
void printTree(Tree *T, int isLogged, char *str_path, char *str_method, FILE *fp_w, int bracket_n, int last);
void printTree_preorder(Tree *T, int isLogged, char *str_path, char *str_method, FILE *fp_w, int bracket_n, int last);

int isLogged = 0;
char *str_logged = "LOGGER";
char *str_logged_method = "-Logged_method";
int main(){
	
	FILE *fp_r;
	FILE *fp_w;
	char bup[MAX_BUP_LEN];
	char str[MAX_BUP_LEN];
	char str_m[100];
	char str_p[1000];
	int ret;
	int i;
	int numOfsnippets = 0;
    int depth;
    int code_snippet_depth = -1;
	TreeNode *parentNode;
	int count_tab = 0;
	int index_str = 0;
	int index_str_p = 0;
    int valid = 0; //It is flag to valid node.
    int path_flag = 0; //It is flag to start building tree.
    int digit = 0;
    int bracket_num = 0;

    int tree_based = 0; //0 : preorder
                        //1 : tree_based
    int depth_flag = 0;

	fp_r = fopen("input/Logging/end2/glassfish-CAST.txt","r");
	fp_w = fopen("output/Logging/end2/glassfish-CAST.json", "w");


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
            continue;
		}else if(strncmp(bup, "path:", 5) == 0){
			if(numOfsnippets != 0){
                if(tree_based){
				    printTree(T, isLogged, str_p, str_m, fp_w, bracket_num,0);
                }else{
                    printTree_preorder(T, isLogged, str_p, str_m, fp_w, bracket_num,0);
                }
				ClearTree(T->root);
				free(T);
				isLogged = 0;

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
            code_snippet_depth = -1;
            bracket_num = 0;
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
            }
        }else{
            continue;
        }
        bracket_num += 2;

        /*  if child num is over 1024, don't insert another node in tree.
        if(check_child_num(T->root) > 0){
            printf("%d\n", check_child_num(T->root));
            continue;
        }*/
        if(path_flag == 1){
            code_snippet_depth = depth;

            if(depth_flag){
                //Node 앞에 depth 한칸 띄어서 추가.
                char temp_str[1000];
                sprintf(temp_str, "%d ", depth-code_snippet_depth+1);
                strcat(temp_str, str);
                strcpy(str, temp_str);
            }

            insert(T, str, NULL, depth);
            path_flag = 0;
        }else{
            if(numOfsnippets != 0){
                if(depth_flag){
                    //Node 앞에 depth 한칸 띄어서 추가.
                    char temp_str[1000];
                    sprintf(temp_str, "%d ", depth-code_snippet_depth+1);
                    strcat(temp_str, str);
                    strcpy(str, temp_str);
                }

                parentNode = searchParentNode(T->root, depth-1);	
                insert(T, str, parentNode, depth);
            }
        }
    }
//	print_preorder(T->root);
    if(tree_based){
	    printTree(T, isLogged, str_p, str_m, fp_w, bracket_num,1);
    }else{
        printTree_preorder(T, isLogged, str_p, str_m, fp_w, bracket_num,1);
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
                return (bup[count_tab]-'0'*100) + (bup[1+count_tab] - '0')*10 + (bup[2+count_tab] - '0');
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

    if(strcmp(bup+digit+count_tab, str_logged) == 0 || strncmp(bup+strlen(bup)-14, str_logged_method, 14) == 0){
        isLogged = 1;
        return 0;
    }else{
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
void printTree(Tree *T, int isLogged, char *str_path, char *str_method, FILE *fp_w,int bracket_n, int last){
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
	fprintf(fp_w, "\t\t\"isLogged\" : %d,\r\n", isLogged);
    fprintf(fp_w, "\t\t\"numOfBracket\" : %d,\r\n", bracket_n);
	fprintf(fp_w, "\t\t\"path\" : \"%s\",\r\n", str_path);
	fprintf(fp_w, "\t\t\"method\" : \"%s\"\r\n", str_method);
	if(last == 0){
		fprintf(fp_w, "\t},\r\n");
	}else{
		fprintf(fp_w, "\t}\r\n");
	}
}
void printTree_preorder(Tree *T, int isLogged, char *str_path, char *str_method, FILE *fp_w, int bracket_n, int last){
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
    fprintf(fp_w, "\t\t\"isLogged\" : %d,\r\n", isLogged);
    fprintf(fp_w, "\t\t\"numOfBracket\" : %d,\r\n", bracket_n);
    fprintf(fp_w, "\t\t\"path\" : \"%s\",\r\n", str_path);
    fprintf(fp_w, "\t\t\"method\" : \"%s\"\r\n", str_method);
    if(last == 0){
        fprintf(fp_w, "\t},\r\n");
    }else{
        fprintf(fp_w, "\t}\r\n");
    }
}
