#include<stdio.h>
#include "add.h"

int main(int argc, char *argv[]) {
    int n = 1000000;
    int res = 0;
    if (launch_and_test(n) == 0) {
        printf("Fail.\n");
        res = 1;
    }
    else 
        printf("Pass.\n");
    if (add_unified(n) == 0) {
        printf("Fail.\n");
        res = 1;
    }
    else 
        printf("Pass.\n");
    return res;
}