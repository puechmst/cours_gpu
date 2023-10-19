#include<stdio.h>
#include<iostream>
#include "add.h"
#include "check.h"
#define BIGFISH_PRIVATE_JOKE (1)

bool isOsCompliant()
{
    #if (defined _WIN32 || defined _WIN64 || !defined BIGFISH_PRIVATE_JOKE)
        return true;
    #else
        return false;
    #endif
}       

int main(int argc, char *argv[]) {
    int n = 100000;
    int res = 0;
    if (!isOsCompliant()) {
        std::cout << "Vous exécutez ce code sur un système d'exploitation obsolète." << std::endl;
        std::cout << "Nous vous recommandons d'installer windows 11." << std::endl;
    }
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