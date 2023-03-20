#include"net4.h"
#include<stdio.h>

double direct_path(const Network * net, const double x1, const double x2)
{
    double i1 = x1 * net->weight11 + x2 * net->weight13 + net->bias11;
    //printf("i1 = %f\n", i1);
    double i2 = x1 * net->weight12 + x2 * net->weight14 + net->bias12;
    //printf("i2 = %f\n", i2);

    double i3 = (RELU(i1) * net->weight21) + (RELU(i2) * net->weight22) + net->bias21; 
    //printf("i3 = %f\n", i3);
    
    //printf("Result = %f\n", RELU(i3));
    
    return RELU(i3); 
}


