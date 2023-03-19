#include"net3.h"

#define LEARNING_RATE 0.01 
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
    

    printf("\nLoss function before training L = %f\n", compute_loss_function(&net, trainingSet, SAMPLES));

    printf("Derivative with respect to weight before training = %f \n", compute_derivative_dl_dw1(&net, trainingSet, SAMPLES));
    printf("Derivative with respect to bias before training = %f \n", compute_derivative_dl_dw0(&net, trainingSet, SAMPLES));

    gradient_descent_once(&net, trainingSet, SAMPLES, LEARNING_RATE);
    printf("\nLoss function after 1 iteration of gradient descent L = %f\n", compute_loss_function(&net, trainingSet, SAMPLES));
    printf("Derivative with respect to weight after 1 iteration of gradient descent = %f \n", compute_derivative_dl_dw1(&net, trainingSet, SAMPLES));
    printf("Derivative with respect to bias after 1 iteration of gradient descent = %f \n", compute_derivative_dl_dw0(&net, trainingSet, SAMPLES)); 
    printf("Weight after 1 iteration of gradient descent w1 = %f\n", net.weight);
    printf("Bias after 1 iterations of gradient descent w1 = %f\n", net.bias);       

    gradient_descent_once(&net, trainingSet, SAMPLES, LEARNING_RATE);
    printf("\nLoss function after 2 iterations of gradient descent L = %f\n", compute_loss_function(&net, trainingSet, SAMPLES));
    printf("Derivative with respect to weight after 2 iterations of gradient descent = %f \n", compute_derivative_dl_dw1(&net, trainingSet, SAMPLES));
    printf("Derivative with respect to bias after 2 iterations of gradient descent = %f \n", compute_derivative_dl_dw0(&net, trainingSet, SAMPLES)); 
    printf("Weight after 2 iterations of gradient descent w1 = %f\n", net.weight);
    printf("Bias after 2 iterations of gradient descent w1 = %f\n", net.bias);    

    gradient_descent_once(&net, trainingSet, SAMPLES, LEARNING_RATE);
    printf("\nLoss function after 3 iterations of gradient descent L = %f\n", compute_loss_function(&net, trainingSet, SAMPLES));
    printf("Derivative with respect to weight after 3 iterations of gradient descent = %f \n", compute_derivative_dl_dw1(&net, trainingSet, SAMPLES));
    printf("Derivative with respect to bias after 3 iterations of gradient descent = %f \n", compute_derivative_dl_dw0(&net, trainingSet, SAMPLES));    
    printf("Weight after 3 iterations of gradient descent w1 = %f\n", net.weight);
    printf("Bias after 3 iterations of gradient descent w1 = %f\n", net.bias);

    gradient_descent_once(&net, trainingSet, SAMPLES, LEARNING_RATE);
    printf("\nLoss function after 4 iterations of gradient descent L = %f\n", compute_loss_function(&net, trainingSet, SAMPLES));
    printf("Derivative with respect to weight after 4 iterations of gradient descent = %f \n", compute_derivative_dl_dw1(&net, trainingSet, SAMPLES));
    printf("Derivative with respect to bias after 4 iterations of gradient descent = %f \n", compute_derivative_dl_dw0(&net, trainingSet, SAMPLES));    
    printf("Weight after 4 iterations of gradient descent w1 = %f\n", net.weight);
    printf("Bias after 4 iterations of gradient descent w1 = %f\n", net.bias);    

    gradient_descent_once(&net, trainingSet, SAMPLES, LEARNING_RATE);
    printf("\nLoss function after 5 iterations of gradient descent L = %f\n", compute_loss_function(&net, trainingSet, SAMPLES));
    printf("Derivative with respect to weight after 5 iterations of gradient descent = %f \n", compute_derivative_dl_dw1(&net, trainingSet, SAMPLES));
    printf("Derivative with respect to bias after 5 iterations of gradient descent = %f \n", compute_derivative_dl_dw0(&net, trainingSet, SAMPLES));    
    printf("Weight after 5 iterations of gradient descent w1 = %f\n", net.weight);
    printf("Bias after 5 iterations of gradient descent w1 = %f\n", net.bias);  

    gradient_descent_once(&net, trainingSet, SAMPLES, LEARNING_RATE);
    printf("\nLoss function after 6 iterations of gradient descent L = %f\n", compute_loss_function(&net, trainingSet, SAMPLES));
    printf("Derivative with respect to weight after 6 iterations of gradient descent = %f \n", compute_derivative_dl_dw1(&net, trainingSet, SAMPLES));
    printf("Derivative with respect to bias after 6 iterations of gradient descent = %f \n", compute_derivative_dl_dw0(&net, trainingSet, SAMPLES));    
    printf("Weight after 6 iterations of gradient descent w1 = %f\n", net.weight);
    printf("Bias after 6 iterations of gradient descent w1 = %f\n", net.bias);  

    gradient_descent_once(&net, trainingSet, SAMPLES, LEARNING_RATE);
    printf("\nLoss function after 7 iterations of gradient descent L = %f\n", compute_loss_function(&net, trainingSet, SAMPLES));
    printf("Derivative with respect to weight after 7 iterations of gradient descent = %f \n", compute_derivative_dl_dw1(&net, trainingSet, SAMPLES));
    printf("Derivative with respect to bias after 7 iterations of gradient descent = %f \n", compute_derivative_dl_dw0(&net, trainingSet, SAMPLES));    
    printf("Weight after 7 iterations of gradient descent w1 = %f\n", net.weight);
    printf("Bias after 7 iterations of gradient descent w1 = %f\n", net.bias);  

gradient_descent_once(&net, trainingSet, SAMPLES, LEARNING_RATE);
    printf("\nLoss function after 8 iterations of gradient descent L = %f\n", compute_loss_function(&net, trainingSet, SAMPLES));
    printf("Derivative with respect to weight after 8 iterations of gradient descent = %f \n", compute_derivative_dl_dw1(&net, trainingSet, SAMPLES));
    printf("Derivative with respect to bias after 8 iterations of gradient descent = %f \n", compute_derivative_dl_dw0(&net, trainingSet, SAMPLES));    
    printf("Weight after 8 iterations of gradient descent w1 = %f\n", net.weight);
    printf("Bias after 8 iterations of gradient descent w1 = %f\n", net.bias);    

    gradient_descent_once(&net, trainingSet, SAMPLES, LEARNING_RATE);
    printf("\nLoss function after 9 iterations of gradient descent L = %f\n", compute_loss_function(&net, trainingSet, SAMPLES));
    printf("Derivative with respect to weight after 9 iterations of gradient descent = %f \n", compute_derivative_dl_dw1(&net, trainingSet, SAMPLES));
    printf("Derivative with respect to bias after 9 iterations of gradient descent = %f \n", compute_derivative_dl_dw0(&net, trainingSet, SAMPLES));    
    printf("Weight after 9 iterations of gradient descent w1 = %f\n", net.weight);
    printf("Bias after 9 iterations of gradient descent w1 = %f\n", net.bias);  

    gradient_descent_once(&net, trainingSet, SAMPLES, LEARNING_RATE);
    printf("\nLoss function after 10 iterations of gradient descent L = %f\n", compute_loss_function(&net, trainingSet, SAMPLES));
    printf("Derivative with respect to weight after 10 iterations of gradient descent = %f \n", compute_derivative_dl_dw1(&net, trainingSet, SAMPLES));
    printf("Derivative with respect to bias after 10 iterations of gradient descent = %f \n", compute_derivative_dl_dw0(&net, trainingSet, SAMPLES));    
    printf("Weight after 10 iterations of gradient descent w1 = %f\n", net.weight);
    printf("Bias after 10 iterations of gradient descent w1 = %f\n", net.bias);        


    i = 0;
    while(i < SAMPLES)
    {
        printf("\nPredicted after 10 iterations of gradient descent Y = %f for x = %f\n", direct_path(&net, trainingSet[i][0]), trainingSet[i][0]);
        i++;
    }    

    return 0;
}

//gcc main.c net3.c -lm