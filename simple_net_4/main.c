#include"net4.h"
#include<stdio.h>

#define LEARNING_RATE 0.01 
#define GRADIENT_DESCENT_ITERATIONS 50
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
 
    
    printf("Network parameters before training: %f, %f, %f, %f, %f, %f, %f, %f, %f\n", net.weight11, net.weight12, net.weight13, net.weight14, net.weight21, net.weight22, net.bias11, net.bias12, net.bias21);


    //function to learn y= (x1 + x2)*2
    double trainingSet [SAMPLES][3] = {{0, 0, 0}, {1, 1, 4}, {2, 1, 6}, {1, 2, 6}, {-2, -2, -8}, {3, 3, 12}, {-1, 3, 4}, {-4, 2, -4}};
    
    int i = 0;
    
    printf("\nPredicted before training for y= (x1 + x2)*2:\n");
    while(i < SAMPLES)
    {
        printf("Y = %f for x1 = %f and x2 = %f\n", direct_path(&net, trainingSet[i][0], trainingSet[i][1]), trainingSet[i][0], trainingSet[i][1]);
        i++;
    } 

           

    return 0;
}

//gcc main.c net4.c -lm