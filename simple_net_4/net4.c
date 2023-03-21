#include"net4.h"
#include<stdio.h>

double direct_path(const Network * net, const double x1, const double x2)
{   
    
    double i1 = x1 * net->weight11 + x2 * net->weight13 + net->bias11;
    double i2 = x1 * net->weight12 + x2 * net->weight14 + net->bias12;
    double i3 = (relu(i1) * net->weight21) + (relu(i2) * net->weight22) + net->bias21; 
    
    return relu(i3); 
}

double compute_loss_function (const Network * net, double training_set[][3], int number_of_samples)
{
    int i = 0;
    double sum = 0;

    while(i < number_of_samples)
    {
        sum += pow((direct_path(net, training_set[i][0], training_set[i][1]) - training_set[i][2]), 2);
        i++;
    }

    return sum;
}