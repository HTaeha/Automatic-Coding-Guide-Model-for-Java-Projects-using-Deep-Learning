typedef struct TreeNode TreeNode;
typedef struct Tree Tree;

#define MAX_QUEUE 1000
#define MAX_STACK 1000

typedef enum{false, true} bool;

typedef TreeNode* element;

typedef struct{
	int front, rear;
	element items[MAX_QUEUE];
} Queue;

typedef struct{
	int top;
	element items[MAX_STACK];
} Stack;
typedef struct{
	int top;
	int items[MAX_STACK];
} Stack2;


struct TreeNode{
	char struc[10000];
	TreeNode *parent;
	TreeNode **child;
	int child_num;
	int child_idx;
	int done;
	int depth;
};

struct Tree{
	TreeNode *root;	
};

Tree *NodeCreate();
void insert(Tree *T, char* struc, TreeNode *p, int depth);
void ClearTree(TreeNode *t);
TreeNode *searchParentNode(TreeNode *T, int depth);
void searchLeafNode(TreeNode *T, Queue *q);
int check_child_num(TreeNode *T);
void print_preorder(TreeNode *T);
void InitQueue(Queue *pqueue);
bool IsFull(Queue *pqueue);
bool IsEmpty(Queue *pqueue);
element Peek(Queue *pqueue);
void EnQueue(Queue *pqueue, element item);
void DeQueue(Queue *pqueue);
void InitStack(Stack *pstack);
bool IsEmptyS(Stack *pstack);
bool IsFullS(Stack *pstack);
void Push(Stack *pstack, element item);
element Pop(Stack *pstack);

void InitStack2(Stack2 *pstack);
bool IsEmptyS2(Stack2 *pstack);
bool IsFullS2(Stack2 *pstack);
void Push2(Stack2 *pstack, int item);
int Pop2(Stack2 *pstack);
