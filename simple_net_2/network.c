#include "network.h"
#include<stdlib.h>

void initialize(Layer * layer, int numInputs, int numNeurons)
{
    layer->weights = malloc(numInputs * numNeurons * sizeof(float));
    layer->biases = malloc(numNeurons * sizeof(float));
    layer->outputs = malloc(numNeurons * sizeof(float));
    
    for (int i = 0; i < numInputs * numNeurons; i++) 
    {
        layer->weights[i] = rand() / (float)RAND_MAX;
    }
    
    for (int i = 0; i < numNeurons; i++) 
    {
        layer->biases[i] = rand() / (float)RAND_MAX;
    }    

}

void computeOutput(Layer * layer, float * inputs, int numInputs, int numNeurons)
{
    
}

