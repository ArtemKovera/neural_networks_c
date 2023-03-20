#ifndef NET4_H
#define NET4_H
#include<math.h>


//simple network for regression with 2 inputs and 1 hidden layer with 2 neurons
typedef struct
{
    double weight11;
    double weight12;
    double weight13;
    double weight14;
    double weight21;
    double weight22;
    double bias11;
    double bias12;
    double bias21;
}Network;

#define RELU(INPUT) (INPUT>0) ? INPUT : 0

#define RELU_DERIV(INPUT) (INPUT>0) ? 1.0 : 0

double direct_path(const Network * net, const double x1, const double x2);

double compute_loss_function(const Network * net, double training_set[][3], int number_of_samples);


#endif