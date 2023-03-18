#include<math.h>

typedef struct
{
    double weight;
    double bias;
}Neuron;

double direct_path(const Neuron * net, const double x);

//compute derivative with respect to weight
double compute_derivative_dl_dw1 (Neuron * net, double training_set[][2], int number_of_samples);

//compute derivative with respect to bias
double compute_derivative_dl_dw0 (Neuron * net, double training_set[][2], int number_of_samples);

double compute_loss_function (Neuron * net, double training_set[][2], int number_of_samples);

void gradient_descent_once (Neuron * net, double training_set[][2], int number_of_samples, double learning_rate);

void train_network (Neuron * net, double training_set[][2], int number_of_samples, double learning_rate, int gradient_descent_iterations);






