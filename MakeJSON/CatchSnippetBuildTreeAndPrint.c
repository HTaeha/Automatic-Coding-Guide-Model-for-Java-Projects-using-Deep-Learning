#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "LoggingTree.h"


void printTree_preorder(Tree *T, int isLogged, char *str_path, char *str_method, FILE *fp_w, int last);
void printTree(Tree *T, int isLogged, char *str_path, char *str_method, FILE *fp_w, int last);
int main(){
	
	FILE *fp_r;
	FILE *fp_w;
	char bup[1000];
	char str[1000];
	char str_m[100];
	char str_p[1000];
	int ret;
	int isLogged = 0;
	int i;
	char *str_logged = "LOGGER";
	int numOfsnippets = 0;
	int depth;
	TreeNode *parentNode;
	int count_tab = 0;
	int try_tab = 0;
	int index_str = 0;
	int index_str_p = 0;
	int tree_insert_ok = 0;
	int print_ok = 0;
	char root_str[100];

    int tree_based = 1; //0 : preorder
                        //1 : tree_based

//	fp_r = fopen("input/Sample/Sample-ppt.txt","r");
//	fp_w = fopen("output/Sample/Sample-ppt.json", "w");
	fp_r = fopen("input/Logging/glassfish-catch_snippet_expression++.txt","r");
	fp_w = fopen("output/Logging/glassfish-catch_snippet_expression++.json", "w");


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
		if(bup[count_tab] <= '9' && bup[count_tab] >= '0'){
			if(bup[1+count_tab] <= '9' && bup[1+count_tab] >= '0'){
				depth = (bup[count_tab]-'0')*10 + (bup[1+count_tab] - '0');
				if(strcmp(bup+2+count_tab, str_logged) == 0){
					isLogged = 1;
					continue;
				}else{
					for(i=2+count_tab;i<strlen(bup);i++){
						if(bup[i] == '\\'){
							continue;
						}else{
							str[index_str++] = bup[i];
						}
					}
					str[index_str] = 0;
					index_str = 0;
				}

			}else{
				depth = bup[count_tab] - '0';
				if(strcmp(bup+1+count_tab, str_logged) == 0){
					isLogged = 1;
					continue;
				}else{
					for(i=1+count_tab;i<strlen(bup);i++){
						if(bup[i] == '\\'){
							continue;
						}else{
							str[index_str++] = bup[i];
						}
					}
					str[index_str] = 0;
					index_str = 0;
				}

			}
		}else if(strncmp(bup, "method:", 7) == 0){
			str_m[0] = 0;
			strcat(str_m, bup+8);
		}else if(strncmp(bup, "path:", 5) == 0){
			if(numOfsnippets != 0){
                if(tree_based){
				    printTree(T, isLogged, str_p, str_m, fp_w, 0);
                }else{
				    printTree_preorder(T, isLogged, str_p, str_m, fp_w, 0);
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
			tree_insert_ok = 0;
			print_ok = 0;
			numOfsnippets++;
		}else{
			continue;
		} 

		if(tree_insert_ok == 0 && strncmp(bup+count_tab+1, "TRY", 3) == 0){
			if(print_ok == 1){
				if(tree_based){
				    printTree(T, isLogged, str_p, str_m, fp_w, 0);
                }else{
				    printTree_preorder(T, isLogged, str_p, str_m, fp_w, 0);
                }
				ClearTree(T->root);
				free(T);
				isLogged = 0;
				Tree *T = NodeCreate();
				insert(T, root_str, NULL, 0);
			}
			tree_insert_ok = 1;
			try_tab = count_tab;
			insert(T, str, T->root, depth);
			continue;
		}

		if(bup[0] == '0'){
			root_str[0] = 0;
			strcpy(root_str, str);
			insert(T, str, NULL, depth);
		}else if(bup[count_tab] <= '9' && bup[count_tab] >= '1'){
			if(tree_insert_ok == 1 && try_tab < depth){
				parentNode = searchParentNode(T->root, depth-1);	
				insert(T, str, parentNode, depth);
			}else if(try_tab >= depth && tree_insert_ok == 1){
				if(strncmp(bup+count_tab+1, "TRY", 3) == 0){
                    if(tree_based){
                        printTree(T, isLogged, str_p, str_m, fp_w, 0);
                    }else{
                        printTree_preorder(T, isLogged, str_p, str_m, fp_w, 0);
                    }

					ClearTree(T->root);
					free(T);
					isLogged = 0;
					try_tab = count_tab;
					numOfsnippets++;
					Tree *T = NodeCreate();
					insert(T, root_str, NULL, 0);
					insert(T, str, T->root, depth);
				}else if(strncmp(bup+count_tab+1, "CATCH", 5) == 0){
					parentNode = searchParentNode(T->root, depth-1);	
					insert(T, str, parentNode, depth);
				}else if(strncmp(bup+count_tab+1, "FINALLY", 7) == 0){
					parentNode = searchParentNode(T->root, depth-1);	
					insert(T, str, parentNode, depth);
				}else{
					tree_insert_ok = 0;
					print_ok = 1;
				}
			}
		}
	}
//	print_preorder(T->root);
    if(tree_based){
        printTree(T, isLogged, str_p, str_m, fp_w, 1);
    }else{
        printTree_preorder(T, isLogged, str_p, str_m, fp_w, 1);
    }

	ClearTree(T->root);
	free(T);

	fprintf(fp_w, "]");

	fclose(fp_r);
	fclose(fp_w);

	return 0;
}
void printTree(Tree *T, int isLogged, char *str_path, char *str_method, FILE *fp_w, int last){
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
	fprintf(fp_w, "\t\t\"path\" : \"%s\",\r\n", str_path);
	fprintf(fp_w, "\t\t\"method\" : \"%s\"\r\n", str_method);
	if(last == 0){
		fprintf(fp_w, "\t},\r\n");
	}else{
		fprintf(fp_w, "\t}\r\n");
	}
}
void printTree_preorder(Tree *T, int isLogged, char *str_path, char *str_method, FILE *fp_w, int last){
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
	fprintf(fp_w, "\t\t\"path\" : \"%s\",\r\n", str_path);
	fprintf(fp_w, "\t\t\"method\" : \"%s\"\r\n", str_method);
	if(last == 0){
		fprintf(fp_w, "\t},\r\n");
	}else{
		fprintf(fp_w, "\t}\r\n");
	}
}
