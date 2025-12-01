#include<stdio.h>
#include<iostream>
#include "add.h"
#include "check.h"


int main(int argc, char *argv[]) {
    int n = 100000;
    int res = 0;
   
    dumpCharacteristics();
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
    close_and_reset();
    return res;
}