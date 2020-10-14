#include "LoggingTree.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

Tree *NodeCreate(){
	Tree *T = (Tree*)malloc(sizeof(Tree));
	
	T->root = NULL;

	return T;
}
void insert(Tree *T, char* struc, TreeNode *p, int depth){
	TreeNode *node = (TreeNode*)malloc(sizeof(TreeNode));
	int i;

	TreeNode **child = (TreeNode**)malloc(sizeof(TreeNode*));
	node->struc[0] = 0;
	strcat(node->struc, struc);
	node->parent = p;
	node->child = child;
	node->child[0] = NULL;
	node->child_num = 1;
	node->child_idx = 0;
	node->done = 0;
	node->depth = depth;

	if(p == NULL){
		T->root = node;
	}else{
		if(p->child_idx == p->child_num){
			p->child_num *= 2;
			p->child = (TreeNode**)realloc(p->child, sizeof(TreeNode*)*(p->child_num));
			if(p->child == NULL){
				printf("realloc failed!!");
				exit(1);
			}
		}
		p->child[p->child_idx++] = node;
	}
}
void ClearTree(TreeNode *t){
	int i;

	if(t != NULL){
		for(i=0;i<t->child_idx;i++){
			ClearTree(t->child[i]);
		}
	}
	free(t->child);
	free(t);
}
TreeNode *searchParentNode(TreeNode *T, int depth){
	int i;

	if(depth == -1){
		return NULL;
	}
	while(1){
		if(T->child_idx != 0){
			T = T->child[(T->child_idx) - 1];
		}else{
			return T;
		}
		if(T->depth > depth){
			return T->parent;
		}
	}
}
void searchLeafNode(TreeNode *T, Queue *q){
	int i;
	if(T != NULL){
		if(T->child_idx == 0){
			EnQueue(q, T);
		}else{
			for(i=0;i<T->child_idx;i++){
				searchLeafNode(T->child[i], q);
			}
		}
	}
}
int check_child_num(TreeNode *T){
    int i;
    if(T != NULL){
        if(T->child_idx >= 1023){
            return 1;
        }else{
            for(i=0;i<T->child_idx;i++){
                return check_child_num(T->child[i]);
            }
        }
    }
    return 0;
}
void print_preorder(TreeNode *T){
	int i;

	if(T != NULL){
		printf("%s\n", T->struc);
		for(i=0;i<T->child_idx;i++){
			print_preorder(T->child[i]);
		}
	}
}
void InitQueue(Queue *pqueue){
	pqueue->front = pqueue->rear = 0;
}
bool IsFull(Queue * pqueue){
	return pqueue->front == (pqueue->rear + 1) % MAX_QUEUE;
}
bool IsEmpty(Queue *pqueue){
	return pqueue->front == pqueue->rear;
}
element Peek(Queue *pqueue){
	if(IsEmpty(pqueue)){
        printf("Queue Peek Error!!\n");
		exit(1);
    }
	return pqueue->items[pqueue->front];
}
void EnQueue(Queue *pqueue, element item){
	if(IsFull(pqueue)){
        printf("Queue EnQueue Error!!\n");
		exit(1);
    }
	pqueue->items[pqueue->rear] = item;
	pqueue->rear = (pqueue->rear + 1) % MAX_QUEUE;
}
void DeQueue(Queue *pqueue){
	if(IsEmpty(pqueue)){
        printf("Queue DeQueue Error!!\n");
		exit(1);
    }
	pqueue->front = (pqueue->front + 1) % MAX_QUEUE;
}
void InitStack(Stack *pstack){
	pstack->top = -1;
}
bool IsEmptyS(Stack *pstack){
	return (pstack->top == -1);
}
bool IsFullS(Stack *pstack){
	return (pstack->top == (MAX_STACK - 1));
}
void Push(Stack *pstack, element item){
	if(IsFullS(pstack)){
        printf("Stack Push Error!!\n");
		exit(1);
	}else{
		pstack->items[++(pstack->top)] = item;
	}
}
element Pop(Stack *pstack){
	if(IsEmptyS(pstack)){
        printf("Stack Pop Error!!\n");
		exit(1);
	}else{
		return pstack->items[(pstack->top)--];
	}
}
void InitStack2(Stack2 *pstack){
	pstack->top = -1;
}
bool IsEmptyS2(Stack2 *pstack){
	return (pstack->top == -1);
}
bool IsFullS2(Stack2 *pstack){
	return (pstack->top == (MAX_STACK - 1));
}
void Push2(Stack2 *pstack, int item){
	if(IsFullS2(pstack)){
        printf("Stack2 Push Error!!\n");
		exit(1);
	}else{
		pstack->items[++(pstack->top)] = item;
	}
}
int Pop2(Stack2 *pstack){
	if(IsEmptyS2(pstack)){
        printf("Stack2 Pop Error!!\n");
		exit(1);
	}else{
		return pstack->items[(pstack->top)--];
	}
}
