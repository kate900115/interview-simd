/**
 * @file main.c
 *
 * @mainpage MNIST 1-Layer Neural Network
 *
 * @brief Main characteristics: Only 1 layer (= input layer), no hidden layer.  Feed-forward only.
 * No Sigmoid activation function. No back propagation.\n
 *
 * @details Learning is achieved simply by incrementally updating the connection weights based on the comparison
 * with the desired target output (supervised learning).\n
 *
 * Its performance (success rate) of 85% is far off the state-of-the-art techniques (surprise, surprise) 
 * but close the Yann Lecun's 88% when using only a linear classifier.
 *
 * @see [Simple 1-Layer Neural Network for MNIST Handwriting Recognition](http://mmlind.github.io/Simple_1-Layer_Neural_Network_for_MNIST_Handwriting_Recognition/)
 * @see http://yann.lecun.com/exdb/mnist/
 * @version [Github Project Page](http://github.com/mmlind/mnist-1lnn/)
 * @author [Matt Lind](http://mmlind.github.io)
 * @date July 2015
 *
 */
 
 
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "screen.h"
#include "mnist-utils.h"
#include "mnist-stats.h"
#include "1lnn.h"
#include <immintrin.h>
//#include <smmintrin.h>






/**
 * @details Trains a layer by looping through and training its cells
 * @param l A pointer to the layer that is to be training
 */

void trainLayer(Layer *l){
    
    // open MNIST files
    FILE *imageFile, *labelFile;
    imageFile = openMNISTImageFile(MNIST_TRAINING_SET_IMAGE_FILE_NAME);
    labelFile = openMNISTLabelFile(MNIST_TRAINING_SET_LABEL_FILE_NAME);
    
    // screen output for monitoring progress
    //displayImageFrame(5,5);

    int errCount = 0;

    // for test performance
    time_t startTrainingTime = time(NULL); 
    
    // Loop through all images in the file
    for (int imgCount=0; imgCount<MNIST_MAX_TRAINING_IMAGES; imgCount++){
        
        // display progress
        //displayLoadingProgressTraining(imgCount,3,5);
        // Reading next image and corresponding label
        MNIST_Image img = getImage(imageFile);
        MNIST_Label lbl = getLabel(labelFile);
        // set target ouput of the number displayed in the current image (=label) to 1, all others to 0
        Vector targetOutput;
        targetOutput = getTargetOutput(lbl);
        //displayImage(&img, 6,6);
    
    	double maxOut = 0;
    	int maxInd = 0;
        // loop through all output cells for the given image

	__m256d v4_zero = _mm256_setzero_pd();
	__m256d v4_one = _mm256_set1_pd(1.0);

	#pragma omp parallel for
        for (int i=0; i < NUMBER_OF_OUTPUT_CELLS; i++){
	    double* input = l->cell[i].input; 
	    double* weight = l->cell[i].weight;
	    uint8_t* pixel = img.pixel;
 	    __m256d v4_sum = _mm256_setzero_pd();

	    double result[4];
 	    for (int j=0; j<NUMBER_OF_INPUT_CELLS; j=j+4){
       	    	//input[j] = pixel[j] ? 1 : 0;
		__m256d v4_pixel = _mm256_set_pd(pixel[j+3], pixel[j+2], pixel[j+1], pixel[j]);
		__m256d v4_input = _mm256_cmp_pd(v4_pixel, v4_zero, 4);
		v4_input = _mm256_and_pd(v4_input, v4_one);
	
//		_mm256_store_pd(result, v4_input);
//		printf("%f\n",result[1]);
		__m256d v4_weight = _mm256_load_pd(weight+j);
//		__m256d v4_weight = _mm256_set_pd(weight[j], weight[j+1], weight[j+2], weight[j+3]);
		__m256d v4_mul = _mm256_mul_pd(v4_weight, v4_input);
		v4_sum = _mm256_add_pd(v4_mul, v4_sum); 
    	    }
	    _mm256_store_pd(result, v4_sum);
	    double c_output_test = result[0] + result[1]+result[2]+result[3];
	
	    l->cell[i].output = c_output_test/NUMBER_OF_INPUT_CELLS;  
 
 	    double err = targetOutput.val[i] - l->cell[i].output;
    	    double temp = err * LEARNING_RATE;

	    __m256d v4_temp = _mm256_set1_pd(temp);

	    double aaa[784];
   
	    for (int j=0; j<NUMBER_OF_INPUT_CELLS; j=j+4){
		__m256d v4_pixel = _mm256_set_pd(pixel[j+3], pixel[j+2], pixel[j+1], pixel[j+0]);
		__m256d v4_input = _mm256_cmp_pd(v4_pixel, v4_zero, 4);
		v4_input = _mm256_and_pd(v4_input, v4_one);
		__m256d v4_mul = _mm256_mul_pd(v4_input, v4_temp);
		__m256d v4_weight = _mm256_load_pd(weight+j);
	//	__m256d v4_weight = _mm256_set_pd(weight[j], weight[j+1], weight[j+2], weight[j+3]);
		v4_weight = _mm256_add_pd(v4_weight, v4_mul);
		
		_mm256_store_pd(aaa+j, v4_weight);

	//	printf("%f, %d\n",weight[j+3], j );
	    }

	    for (int j=0; j<784; j++){
		weight[j]=aaa[j];	
	    }
 /*
    	    #pragma simd
   	    for (int j=0; j<NUMBER_OF_INPUT_CELLS; j++){
		if (pixel[j])
        	weight[j] += temp;
    	    }*/
   
        }
 
        int predictedNum = getLayerPrediction(l);
        if (predictedNum!=lbl) errCount++;
          
        //printf("\n      Prediction: %d   Actual: %d ",predictedNum, lbl);

        //displayProgress(imgCount, errCount, 3, 66);
        
    }

    time_t endTrainingTime = time(NULL);
    double trainingTime = difftime(endTrainingTime, startTrainingTime);
    printf("Training time is %.1f sec\n", trainingTime);
    double successRate = 100.0- (double)errCount/(double)(MNIST_MAX_TRAINING_IMAGES)*100;
    printf("training successful-rate = %.2f%%\n", successRate);
	
    // Close files
    fclose(imageFile);
    fclose(labelFile);

}




/**
 * @details Tests a layer by looping through and testing its cells
 * Exactly the same as TrainLayer() but WITHOUT LEARNING.
 * @param l A pointer to the layer that is to be training
 */

void testLayer(Layer *l){
    
    // open MNIST files
    FILE *imageFile, *labelFile;
    imageFile = openMNISTImageFile(MNIST_TESTING_SET_IMAGE_FILE_NAME);
    labelFile = openMNISTLabelFile(MNIST_TESTING_SET_LABEL_FILE_NAME);
    
    
    // screen output for monitoring progress
    //displayImageFrame(7,5);
    
    int errCount = 0;
   
    time_t startTestingTime = time(NULL);
 
    // Loop through all images in the file
    for (int imgCount=0; imgCount<MNIST_MAX_TESTING_IMAGES; imgCount++){
        
        // display progress
        //displayLoadingProgressTesting(imgCount,5,5);
        
        // Reading next image and corresponding label
        MNIST_Image img = getImage(imageFile);
        MNIST_Label lbl = getLabel(labelFile);
        
        // set target ouput of the number displayed in the current image (=label) to 1, all others to 0
        Vector targetOutput;
        targetOutput = getTargetOutput(lbl);
        
       // displayImage(&img, 8,6);
        
        // loop through all output cells for the given image
	__m256d v4_zero = _mm256_setzero_pd();
	__m256d v4_one = _mm256_set1_pd(1.0);

	#pragma omp parallel for
        for (int i=0; i < NUMBER_OF_OUTPUT_CELLS; i++){
	    double* input = l->cell[i].input; 
	    double* weight = l->cell[i].weight;
	    uint8_t* pixel = img.pixel;
 	    __m256d v4_sum = _mm256_setzero_pd();

	    double result[4];
 	    for (int j=0; j<NUMBER_OF_INPUT_CELLS; j=j+4){
       	    	//input[j] = pixel[j] ? 1 : 0;
		__m256d v4_pixel = _mm256_set_pd(pixel[j+3], pixel[j+2], pixel[j+1], pixel[j]);
		__m256d v4_input = _mm256_cmp_pd(v4_pixel, v4_zero, 4);
		v4_input = _mm256_and_pd(v4_input, v4_one);
		//__m256d v4_weight = _mm256_set_pd(weight[j], weight[j+1], weight[j+2], weight[j+3]);
		__m256d v4_weight = _mm256_load_pd(weight+j);
		__m256d v4_mul = _mm256_mul_pd(v4_weight, v4_input);
		v4_sum = _mm256_add_pd(v4_mul, v4_sum); 
    	    }
	    _mm256_store_pd(result, v4_sum);
	    double c_output_test = result[0] + result[1]+result[2]+result[3];
	
	    l->cell[i].output = c_output_test/NUMBER_OF_INPUT_CELLS;   
        }
        
        int predictedNum = getLayerPrediction(l);
        if (predictedNum!=lbl) errCount++;
        
        //printf("\n      Prediction: %d   Actual: %d ",predictedNum, lbl);
        
        //displayProgress(imgCount, errCount, 5, 66);
      
    }
    
    time_t endTestingTime = time(NULL);
    double testingTime = difftime(endTestingTime, startTestingTime);
    double successRate = 100.0- (double)errCount/(double)(MNIST_MAX_TESTING_IMAGES)*100;
    printf("testing successful-rate = %.2f%%\n", successRate);
	
    printf("testing time is: %.1f sec \n", testingTime);

    // Close files
    fclose(imageFile);
    fclose(labelFile);
    
}





/**
 * @details Main function to run MNIST-1LNN
 */

int main(int argc, const char * argv[]) {
    
    // remember the time in order to calculate processing time at the end
    time_t startTime = time(NULL);
    
    // clear screen of terminal window
    clearScreen();
    printf("    MNIST-1LNN: a simple 1-layer neural network processing the MNIST handwriting images\n");
    
    // initialize all connection weights to random values between 0 and 1
    Layer outputLayer;
    initLayer(&outputLayer);
    trainLayer(&outputLayer);

    printf("Done training\n");
    testLayer(&outputLayer);

    locateCursor(38, 5);
    
    // Calculate and print the program's total execution time
    time_t endTime = time(NULL);
    double executionTime = difftime(endTime, startTime);
    printf("DONE! Total execution time: %.1f sec\n",executionTime);
    
    return 0;
}


