#include"net3.h"
#include<stdio.h>

#define LEARNING_RATE 0.01 
#define GRADIENT_DESCENT_ITERATIONS 50
#define WEIGHT_INITIALIZATION 1.0
#define BIAS_INITIALIZATION 1.0
#define SAMPLES 4
#define X1 -5.0
#define X2 10.0

int main(void)
{
    Neuron net = {.weight=WEIGHT_INITIALIZATION, .bias=BIAS_INITIALIZATION};
    
    //function to learn y=2x
    double trainingSet [SAMPLES][2] = {{-2, -4}, {-1, -2}, {0, 0}, {3, 6}};
    
    int i = 0;
    
    printf("\nPredicted before training for y=2x:\n");
    while(i < SAMPLES)
    {
        printf("Y = %f for x = %f\n", direct_path(&net, trainingSet[i][0]), trainingSet[i][0]);
        i++;
    }

    train_network(&net, trainingSet, SAMPLES, LEARNING_RATE, GRADIENT_DESCENT_ITERATIONS);
    printf("\nWeight after training w1 = %f\n", net.weight);
    printf("Bias after training w0 = %f\n", net.bias);        
    

    printf("\nPredicted after training for y=2x:\n");
     i = 0;
    while(i < SAMPLES)
    {
        printf("Y = %f for x = %f\n", direct_path(&net, trainingSet[i][0]), trainingSet[i][0]);
        i++;
    } 

    printf("Y = %f for x = %f\n", direct_path(&net, X1), X1);  
    printf("Y = %f for x = %f\n", direct_path(&net, X2), X2); 


    printf("\n------------------------------------------\n");

    Neuron net2 = {.weight=WEIGHT_INITIALIZATION, .bias=BIAS_INITIALIZATION};
    
    //function to learn y=-10x
    double trainingSet2 [SAMPLES][2] = {{-2, 20}, {-1, 10}, {0, 0}, {3, -30}};
    
    i = 0;
    
    printf("\nPredicted before training for y=-10x:\n");
    while(i < SAMPLES)
    {
        printf("Y = %f for x = %f\n", direct_path(&net2, trainingSet2[i][0]), trainingSet2[i][0]);
        i++;
    }  

    train_network(&net2, trainingSet2, SAMPLES, LEARNING_RATE, GRADIENT_DESCENT_ITERATIONS);
    printf("\nWeight after training w1 = %f\n", net2.weight);
    printf("Bias after training w0 = %f\n", net2.bias);        
    

   
    printf("\nPredicted after training for y=-10x:\n");
    i = 0;
    while(i < SAMPLES)
    {
        printf("Y = %f for x = %f\n", direct_path(&net2, trainingSet2[i][0]), trainingSet2[i][0]);
        i++;
    } 

    printf("Y = %f for x = %f\n", direct_path(&net2, X1), X1);  
    printf("Y = %f for x = %f\n", direct_path(&net2, X2), X2);       

    return 0;
}

//gcc main.c net3.c -lm