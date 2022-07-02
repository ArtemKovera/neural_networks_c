/* super simple neural network-perceptron with only one input neuron learning linear function */

#include<stdlib.h>
#include<stdio.h>
#include<math.h>

#define ROWS 2
#define COLS 2
#define WEIGHT 0.5
#define BIAS 0.5
#define LEARNING_RATE 0.005
#define MAX_ITERATIONS 30
#define MIN_LOSS 0.002
#define NEW_INPUT 5.0


typedef struct
{
    double weight;
    double bias;
} NeuralNet;


double neuron(const double input, const NeuralNet * const net)
{
    return net->weight * input + net->bias;
}

double loss(NeuralNet * const net, const double trainingSet [][COLS], const int size)
{   
    double sum = 0;

    for(int i = 0; i < size; i++)
    {
        sum += pow((neuron(trainingSet[i][0], net) - trainingSet[i][1]), 2);
    }

    return sum;    
}

void trainNet(NeuralNet * const net, const double trainingSet[][COLS], const int size, const int maxIterations, const double learningRate, const double minLoss)
{
    int i = 0;
    double s0, s1, derivativeWeight, derivativeBias, newWeight, newBias;
    double l = loss(net, trainingSet, size);

    while(i < maxIterations || l > MIN_LOSS)
    {
        s0 = neuron(trainingSet[0][0], net) - trainingSet[0][1];
        s1 = neuron(trainingSet[1][0], net) - trainingSet[1][1];

        derivativeWeight = 2 * trainingSet[0][0] * s0 + 2 * trainingSet[1][0] * s1;
        newWeight = net->weight - learningRate * derivativeWeight;
        
        derivativeBias = 2 * s0 + 2 * s1;
        newBias = net->bias - learningRate * derivativeBias;

        net->weight = newWeight;
        net->bias = newBias;
        
        l = loss(net, trainingSet, size);
        
        //printf("New loss = %f\n", l); 

        i++;
    }   
   
}


int main(void)
{

    //linear relation 2x + 1 
    double trainingSet[ROWS][COLS] = { {2, 5}, {4, 9} }; 
    

    //initialising weight and bias to 0.5
    NeuralNet net = {WEIGHT, BIAS};
    
    printf("Correct value = %f\n", (NEW_INPUT*2+1));
    printf("---- Predicted before training = %f\n", neuron(NEW_INPUT, &net));

    trainNet(&net, trainingSet, ROWS, MAX_ITERATIONS, LEARNING_RATE, MIN_LOSS);

    printf("---- Predicted after training = %f\n", neuron(NEW_INPUT, &net));

    return 0;
}