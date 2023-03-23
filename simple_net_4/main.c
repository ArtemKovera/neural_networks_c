#include"net4.h"
#include<stdio.h>

#define LEARNING_RATE 0.001 
#define GRADIENT_DESCENT_ITERATIONS 400
#define WEIGHT_INITIALIZATION 1.0
#define BIAS_INITIALIZATION 1.0
#define SAMPLES 8
#define X1 -5.0
#define X2 10.0

int main(void)
{
    Network net;

    net.weight11 = net.weight12 = net.weight13 = net.weight14 = net.weight21 = net.weight22 = WEIGHT_INITIALIZATION;
    net.bias11 = net.bias12 = net.bias21 = BIAS_INITIALIZATION;  

    printf("\nChecking RELU: RELU(-3) = %f, RELU(4) = %f, RELU(0) = %f, RELU(1) = %f, RELU(-1) = %f\n", RELU(-3.0), RELU(4.0), RELU(0.0), RELU(1.0), RELU(-1.0));
 
    printf("\nChecking relu(): relu(-3) = %f, relu(4) = %f, relu(0) = %f, relu(1) = %f, relu(-1) = %f\n", relu(-3.0), relu(4.0), relu(0.0), relu(1.0), relu(-1.0));
    
    printf("Network parameters before training: %f, %f, %f, %f, %f, %f, %f, %f, %f\n", net.weight11, net.weight12, net.weight13, net.weight14, net.weight21, net.weight22, net.bias11, net.bias12, net.bias21);


    //function to learn y= (x1 + x2)*2
    double trainingSet [SAMPLES][3] = {{0, 0, 0}, {1, 1, 4}, {2, 1, 6}, {8, 2, 20}, {2, 2, 8}, {3, 3, 12}, {1, 3, 8}, {5, 5, 20}};
    
    int i = 0;
    
    printf("\nPredicted before training for y= (x1 + x2)*2:\n");
    while(i < SAMPLES)
    {
        printf("Y = %f for x1 = %f and x2 = %f\n", direct_path(&net, trainingSet[i][0], trainingSet[i][1]), trainingSet[i][0], trainingSet[i][1]);
        i++;
    } 

    printf("Loss function before training L = %f\n", compute_loss_function(&net, trainingSet, SAMPLES));    

    train_network(&net, trainingSet, SAMPLES, LEARNING_RATE, GRADIENT_DESCENT_ITERATIONS);
    printf("\nNetwork parameters after training: %f, %f, %f, %f, %f, %f, %f, %f, %f\n", net.weight11, net.weight12, net.weight13, net.weight14, net.weight21, net.weight22, net.bias11, net.bias12, net.bias21);
    printf("\nPredicted after training for y= (x1 + x2)*2:\n");
    i = 0;
    while(i < SAMPLES)
    {
        printf("Y = %f for x1 = %f and x2 = %f\n", direct_path(&net, trainingSet[i][0], trainingSet[i][1]), trainingSet[i][0], trainingSet[i][1]);
        i++;
    }     
    printf("Y = %f for x1 = %f and x2 = %f\n", direct_path(&net, 6.0, 4.0), 6.0, 4.0);
    printf("Y = %f for x1 = %f and x2 = %f\n", direct_path(&net, 10.0, 10.0), 10.0, 10.0);


    return 0;
}

//gcc main.c net4.c -lm