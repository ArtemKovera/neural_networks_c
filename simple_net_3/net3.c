#include"net3.h"

double direct_path(const Neuron * net, const double x)
{
    return x * net->weight + net->bias;
}

double compute_loss_function (Neuron * net, double training_set[][2], int number_of_samples)
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

/*
double compute_derivative_dl_dw1 (Neuron * net, double training_set[][2], int number_of_samples)
{

}
*/