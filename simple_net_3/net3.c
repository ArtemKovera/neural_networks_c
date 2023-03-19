#include"net3.h"

double direct_path(const Neuron * net, const double x)
{
    return x * net->weight + net->bias;
}

double compute_loss_function (const Neuron * net, double training_set[][2], int number_of_samples)
{
    int i = 0;
    double sum;

    while(i < number_of_samples)
    {
        sum += pow((direct_path(net, training_set[i][0]) - training_set[i][1]), 2);
        i++;
    }

    return sum;
}

double compute_derivative_dl_dw1 ( const Neuron * net, double training_set [][2], int number_of_samples )
{
    int i = 0;;
    double sum = 0;
    
    while(i < number_of_samples)
    {
        sum += (2 * training_set[i][0] * (direct_path(net, training_set[i][0]) - training_set[i][1]));
        i++;
    }
    
    return sum;
}

double compute_derivative_dl_dw0 ( const Neuron * net, double training_set [][2], int number_of_samples )
{
    int i = 0;;
    double sum = 0;
    
    while(i < number_of_samples)
    {
        sum += (2 * (direct_path(net, training_set[i][0]) - training_set[i][1]));
        i++;
    }

    return sum;
}

void gradient_descent_once (Neuron * net, double training_set[][2], int number_of_samples, double learning_rate)
{
    net->weight -= compute_derivative_dl_dw1(net, training_set, number_of_samples) * learning_rate;
    net->bias   -= compute_derivative_dl_dw0(net, training_set, number_of_samples) * learning_rate;    
}

void train_network (Neuron * net, double training_set[][2], int number_of_samples, double learning_rate, int gradient_descent_iterations)
{
    int i = 0;

    while(i < gradient_descent_iterations)
    {
        compute_loss_function(net, training_set, number_of_samples);
        gradient_descent_once(net, training_set, number_of_samples, learning_rate);
        i++;
    }    
}