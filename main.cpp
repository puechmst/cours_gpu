#include<stdio.h>
#include "add.h"

int main(int argc, char *argv[]) {
    int n = 1000000;
    if (launch_and_test(n) == 0)
        printf("Echec.\n");
    else 
        printf("Vérifié.\n");
    if (add_unified(n) == 0)
        printf("Echec.\n");
    else 
        printf("Vérifié.\n");
}