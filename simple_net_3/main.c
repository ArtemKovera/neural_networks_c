#include"net3.h"

#define LEARNING_RATE 0.1 
#define GRADIENT_DESCENT_ITERATIONS 20
#define WEIGHT_INITIALIZATION 1.0
#define BIAS_INITIALIZATION 1.0
#define SAMPLES 4
#define X 2.0

int main(void)
{
    Neuron net = {.weight=WEIGHT_INITIALIZATION, .bias=BIAS_INITIALIZATION};
    
    //function to learn y=2x
    double trainingSet [SAMPLES][2] = {{-2, -4}, {-1, -2}, {0, 0}, {3, 6}};
    
    int i = 0;

    while(i < SAMPLES)
    {
        printf("Predicted before training Y = %f for x = %f\n", direct_path(&net, trainingSet[i][0]), trainingSet[i][0]);
        i++;
    }
    

    printf("Loss function before training L = %f\n", compute_loss_function(&net, trainingSet, SAMPLES));

    return 0;
}

//gcc main.c net3.c -lm