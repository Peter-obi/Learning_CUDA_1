#include "book.h"

#define N 10

void add(int *a, int *b, int *c){
    for (int i = 0; i < N; i++){    
         c[i] = a[i] + b[i];
    }  
}

int main(void){

    int a[N], b[N], c[N];

    for (int i = 0; i < N; i++){
        a[i] = -i;
        b[i] = i * i;
    }

    add(a, b, c);

    for(int i =0; i  < N; i++){
        printf("%d + %d = %d\n", a[i], b[i], c[i]);
    }

    return 0;
}
