#ifndef NETWORK_H
#define NETWORK_H

#include<math.h>


#define sigmoid(x) = 1 / (1 + exp(-1 * (x)))

typedef struct
{
    float * weights;
    float * biases;
    float * outputs;
    
}Layer;

//function to initialize weights and biases and allocating memory for weights, biases, and outputs of a single layer
void initialize(Layer * layer, int numInputs, int numNeurons);

//compute the output of a layer
void computeOutput(Layer * layer, float * inputs, int numInputs, int numNeurons); 


#endif