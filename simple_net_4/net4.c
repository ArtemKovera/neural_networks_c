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

void gradient_descent_once (Network * net, double training_set[][3], int number_of_samples, double learning_rate)
{
    int i = 0;
    double f1, f2, f3;
    double common_term; 
    double sum = 0;

    //derivatives with respect to parameters
    double df_db3 = 0, df_dw21 = 0, df_dw22 = 0, df_dw11 = 0, df_dw12 = 0, df_db1 = 0, df_dw13 = 0, df_dw14 = 0, df_db2 = 0;

    while(i < number_of_samples)
    {
        f1 = direct_path(net, training_set[i][0], training_set[i][1]);
        f2 = relu(net->weight11 * training_set[i][0] + net->weight12 * training_set[i][1] + net->bias11);
        f3 = relu(net->weight13 * training_set[i][0] + net->weight14 * training_set[i][1] + net->bias12);
        common_term = (f1 - training_set[i][2]) * 2 * relu_der(f1);

        //computing derivatives 
        df_db3 += common_term;
        df_dw21 += common_term * f2;
        df_dw22 += common_term * f3;
        df_dw11 += common_term * net->weight21 * relu_der(f2) * training_set[i][0];
        df_dw12 += common_term * net->weight21 * relu_der(f2) * training_set[i][1];
        df_dw13 += common_term * net->weight22 * relu_der(f3) * training_set[i][0];
        df_dw14 += common_term * net->weight22 * relu_der(f3) * training_set[i][1];
        df_db1 += common_term * net->weight21 * relu_der(f2);
        df_db2 += common_term * net->weight22 * relu_der(f3);   

        i++;
    }  

    //gradient descent
    net->weight11 -= df_dw11 * learning_rate;
    net->weight12 -= df_dw12 * learning_rate;
    net->weight13 -= df_dw13 * learning_rate;
    net->weight14 -= df_dw14 * learning_rate;
    net->weight21 -= df_dw21 * learning_rate;
    net->weight22 -= df_dw21 * learning_rate;
    net->bias11 -= df_db1 * learning_rate;
    net->bias12 -= df_db2 * learning_rate;
    net->bias21 -= df_db3 * learning_rate;

}

void train_network (Network * net, double training_set[][3], int number_of_samples, double learning_rate, int gradient_descent_iterations)
{
    int i = 0;

    while(i < gradient_descent_iterations)
    {
        gradient_descent_once(net, training_set, number_of_samples, learning_rate);
        printf("Loss function after %d iterations of gradient descent L = %f\n", i, compute_loss_function(net, training_set, number_of_samples));
        i++;
    }    
}    