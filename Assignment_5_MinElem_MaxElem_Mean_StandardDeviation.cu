%%cu
//Author : Piyush Rajendra Chaudhari
//Roll No: BECOC311
//Division : C
//My GitHub Repository Code Link :
//https://github.com/piyush26c/Cuda-Programming/blob/main/Assignment_5_MinElem_MaxElem_Mean_StandardDeviation.cu
//Code Title : Find Min, Max, Sum, Mean, Standard Deviation from a random generated vector, by using parallel reduction.
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <math.h>
#include <time.h>
#include <limits.h>

#define VECTOR_SIZE (1 << (VECTOR_SIZE_))   //corresponds to 2^VECTOR_SIZE_
#define VECTOR_SIZE_ 3

void fillVector(long long *vector_) {
    srand(time(0));
    for (int indx = 0; indx < VECTOR_SIZE; indx++) {
        vector_[indx] = rand() % 1000;
    }
}

void fillVectorWDV(double *vector_) {
    srand(time(0));
    for (int indx = 0; indx < VECTOR_SIZE; indx++) {
        vector_[indx] = (double) (indx + 1) * (rand() % 1000);
    }
}

void printVector(long long *vector_) {
    for (int indx = 0; indx < VECTOR_SIZE; indx++) {
        printf("%lld ", vector_[indx]);
    }
}

void printVectorWDV(double *vector_) {
    for (int indx = 0; indx < VECTOR_SIZE; indx++) {
        printf("%lf ", vector_[indx]);
    }
}

__global__ void findMinElem(long long *vector_) {
    int threadId = threadIdx.x;
    long long stepSize = 1;
    int numberOfThreads = blockDim.x;
    while (numberOfThreads > 0) {
        if (threadId < numberOfThreads) {
            int firstElemIndx = threadId * stepSize * 2;
            int secondElemIndx = firstElemIndx + stepSize;
            if (vector_[firstElemIndx] > vector_[secondElemIndx]) {
                vector_[firstElemIndx] = vector_[secondElemIndx];
            }
        }
        stepSize = stepSize * 2;
        numberOfThreads = numberOfThreads / 2;
    }
}

__global__ void findMaxElem(long long *vector_) {

    int threadId = threadIdx.x;
    long long stepSize = 1;
    int numberOfThreads = blockDim.x;
    while (numberOfThreads > 0) {
        if (threadId < numberOfThreads) {
            int firstElemIndx = threadId * stepSize * 2;
            int secondElemIndx = firstElemIndx + stepSize;
            if (vector_[firstElemIndx] < vector_[secondElemIndx]) {
                vector_[firstElemIndx] = vector_[secondElemIndx];
            }
        }
        stepSize = stepSize * 2;
        numberOfThreads = numberOfThreads / 2;
    }
}

__global__ void findMean(double *vector_) {

    int threadId = threadIdx.x;
    long long stepSize = 1;
    int numberOfThreads = blockDim.x;
    while (numberOfThreads > 0) {
        if (threadId < numberOfThreads) {
            int firstElemIndx = threadId * stepSize * 2;
            int secondElemIndx = firstElemIndx + stepSize;
            vector_[firstElemIndx] += vector_[secondElemIndx];
        }
        stepSize = stepSize * 2;
        numberOfThreads = numberOfThreads / 2;
    }
    vector_[0] = vector_[0] / (double) VECTOR_SIZE;
}

__global__ void findSum(long long *vector_) {

    int threadId = threadIdx.x;
    long long stepSize = 1;
    int numberOfThreads = blockDim.x;
    while (numberOfThreads > 0) {
        if (threadId < numberOfThreads) {
            int firstElemIndx = threadId * stepSize * 2;
            int secondElemIndx = firstElemIndx + stepSize;
            vector_[firstElemIndx] += vector_[secondElemIndx];
        }
        stepSize = stepSize * 2;
        numberOfThreads = numberOfThreads / 2;
    }
}

__global__ void findSumVar(double *vector_) {

    int threadId = threadIdx.x;
    long long stepSize = 1;
    int numberOfThreads = blockDim.x;
    while (numberOfThreads > 0) {
        if (threadId < numberOfThreads) {
            int firstElemIndx = threadId * stepSize * 2;
            int secondElemIndx = firstElemIndx + stepSize;
            vector_[firstElemIndx] += vector_[secondElemIndx];
        }
        stepSize = stepSize * 2;
        numberOfThreads = numberOfThreads / 2;
    }
}

__global__ void findVariance(double *vector_, double mean) {

    int threadId = threadIdx.x;
    //printf("\nthreaad : %d, vector_[threadid] : %lf", threadId, vector_[threadId]);
    vector_[threadId] = pow((vector_[threadId] - mean), 2);
}

long long findMinElemHost(long long *vector_) {
    long long ans = LONG_MAX;
    for (int i = 0; i < VECTOR_SIZE; i++) {
        if (vector_[i] < ans) {
            ans = vector_[i];
        }
    }
    return ans;
}

int main(void) {
    {
        //program for finding minimum element from input vector
        long long *hostVector, resultMinElem;
        long long *deviceVector;
        long long memorySize = VECTOR_SIZE * sizeof(long long);

        // Allocate space for host vectors A and insert input values
        hostVector = (long long *) malloc(memorySize);
        fillVector(hostVector);

        // Allocate space for device vectors A
        cudaMalloc((void **) &deviceVector, memorySize);

        // Copy vector data from host to device
        cudaMemcpy(deviceVector, hostVector, memorySize, cudaMemcpyHostToDevice);


        dim3 blocksPerGrid(1, 1, 1);
        //by creating single block with half of the size of vector threads.
        dim3 threadsPerBlock(VECTOR_SIZE / 2, 1, 1);
        float gpu_ElapseTime = 0;
        cudaEvent_t start, stop;

        (cudaEventCreate(&start));
        (cudaEventCreate(&stop));

        (cudaEventRecord(start, 0));
        findMinElem << < blocksPerGrid, threadsPerBlock >> > (deviceVector);
        (cudaEventRecord(stop, 0));
        (cudaEventSynchronize(stop));

        (cudaEventElapsedTime(&gpu_ElapseTime, start, stop));

        (cudaEventDestroy(start));
        (cudaEventDestroy(stop));

        // Copy result back to host result variable
        cudaMemcpy(&resultMinElem, deviceVector, sizeof(long long), cudaMemcpyDeviceToHost);

        clock_t cpu_startTime, cpu_endTime;
        double cpu_ElapseTime = 0;
        cpu_startTime = clock();
        long long ansFromHost = findMinElemHost(hostVector);
        cpu_endTime = clock();
        cpu_ElapseTime = (cpu_endTime - cpu_startTime);
        printf("\n\nOutput : ");
        printf("\nInput Vector-1 : ");
        printVector(hostVector);
        printf("\n\nThe minimum element in given vector-1 using GPU : %lld", resultMinElem);
        printf("\nThe elapsed time in GPU                         : %.6f ms", gpu_ElapseTime);
        printf("\n\nThe minimum element in given vector-1 using CPU : %lld", resultMinElem);
        printf("\nThe elapsed time in CPU                         : %.6lf ms", cpu_ElapseTime - gpu_ElapseTime);

    }
    {
        //program for finding maximum element from input vector

        long long *hostVector, resultMaxElem;
        long long *deviceVector;
        long long memorySize = VECTOR_SIZE * sizeof(long long);

        // Allocate space for host vectors A and insert input values
        hostVector = (long long *) malloc(memorySize);
        fillVector(hostVector);

        // Allocate space for device vectors A
        cudaMalloc((void **) &deviceVector, memorySize);

        // Copy vector data from host to device
        cudaMemcpy(deviceVector, hostVector, memorySize, cudaMemcpyHostToDevice);


        dim3 blocksPerGrid(1, 1, 1);
        //by creating single block with half of the size of vector threads.
        dim3 threadsPerBlock(VECTOR_SIZE / 2, 1, 1);
        findMaxElem <<< blocksPerGrid, threadsPerBlock >>> (deviceVector);

        // Copy result back to host result variable
        cudaMemcpy(&resultMaxElem, deviceVector, sizeof(long long), cudaMemcpyDeviceToHost);

        //printf("\nInput Vector-1 : ");
        //printVector(hostVector);
        printf("\n\nThe maximum element in given vector-1           : %lld", resultMaxElem);
    }
    {
        //program for finding mean from input vector

        double *hostVector;
        double resultMean;
        double *deviceVector;
        double memorySize = VECTOR_SIZE * sizeof(double);

        // Allocate space for host vectors A and insert input values
        hostVector = (double *) malloc(memorySize);
        fillVectorWDV(hostVector);

        // Allocate space for device vectors A
        cudaMalloc((void **) &deviceVector, memorySize);

        // Copy vector data from host to device
        cudaMemcpy(deviceVector, hostVector, memorySize, cudaMemcpyHostToDevice);


        dim3 blocksPerGrid(1, 1, 1);
        //by creating single block with half of the size of vector threads.
        dim3 threadsPerBlock(VECTOR_SIZE / 2, 1, 1);
        findMean <<< blocksPerGrid, threadsPerBlock >>> (deviceVector);

        // Copy result back to host result variable
        cudaMemcpy(&resultMean, deviceVector, sizeof(double), cudaMemcpyDeviceToHost);
        printf("\n\nInput Vector-2 (with double datatype values) : ");
        printVectorWDV(hostVector);
        printf("\n\nThe mean of the given vector-2                           : %lf", resultMean);
    }
    {
        //program for finding sum of all elements from input vector

        long long *hostVector, resultSum;
        long long *deviceVector;
        long long memorySize = VECTOR_SIZE * sizeof(long long);

        // Allocate space for host vectors A and insert input values
        hostVector = (long long *) malloc(memorySize);
        fillVector(hostVector);

        // Allocate space for device vectors A
        cudaMalloc((void **) &deviceVector, memorySize);

        // Copy vector data from host to device
        cudaMemcpy(deviceVector, hostVector, memorySize, cudaMemcpyHostToDevice);


        dim3 blocksPerGrid(1, 1, 1);
        //by creating single block with half of the size of vector threads.
        dim3 threadsPerBlock(VECTOR_SIZE / 2, 1, 1);
        findSum <<< blocksPerGrid, threadsPerBlock >>> (deviceVector);

        // Copy result back to host result variable
        cudaMemcpy(&resultSum, deviceVector, sizeof(long long), cudaMemcpyDeviceToHost);

        printf("\nThe summation of all the elements in given vector-2      : %lld", resultSum);
    }
    {
        //program for finding standard deviation of elements from input vector
        //first find mean and then substract it from each element parallelly

        //program for finding mean from input vector

        double *hostVector;
        double resultMean;
        double *deviceVector;
        double memorySize = VECTOR_SIZE * sizeof(double);

        // Allocate space for host vectors A and insert input values
        hostVector = (double *) malloc(memorySize);
        fillVectorWDV(hostVector);

        // Allocate space for device vectors A
        cudaMalloc((void **) &deviceVector, memorySize);

        // Copy vector data from host to device
        cudaMemcpy(deviceVector, hostVector, memorySize, cudaMemcpyHostToDevice);


        dim3 blocksPerGrid(1, 1, 1);
        //by creating single block with half of the size of vector threads.
        dim3 threadsPerBlock(VECTOR_SIZE / 2, 1, 1);
        findMean <<< blocksPerGrid, threadsPerBlock >>> (deviceVector);

        // Copy result back to host result variable
        cudaMemcpy(&resultMean, deviceVector, sizeof(double), cudaMemcpyDeviceToHost);
        cudaFree(deviceVector);

        //again,...
        // Allocate space for device vectors A
        cudaMalloc((void **) &deviceVector, memorySize);
        double *diffSumSqrVector;
        diffSumSqrVector = (double *) malloc(memorySize);

        // Copy vector data from host to device
        cudaMemcpy(deviceVector, hostVector, memorySize, cudaMemcpyHostToDevice);

        findVariance <<< 1, VECTOR_SIZE >>> (deviceVector, resultMean);
        //diffSumSqrVector stores (x(i) - xmean)^2 at each index, further we have to sum up aall and divide with VECTOR_SIZE to get variance.
        cudaMemcpy(diffSumSqrVector, deviceVector, memorySize, cudaMemcpyDeviceToHost);
        cudaFree(deviceVector);

        double resultVarSum;
        //again,...
        // Allocate space for device vectors A
        cudaMalloc((void **) &deviceVector, memorySize);

        // Copy vector data from host to device
        cudaMemcpy(deviceVector, diffSumSqrVector, memorySize, cudaMemcpyHostToDevice);

        findSumVar <<< 1, VECTOR_SIZE >>> (deviceVector);
        // Copy result back to host result variable
        cudaMemcpy(&resultVarSum, deviceVector, sizeof(double), cudaMemcpyDeviceToHost);
        cudaFree(deviceVector);

        double variance = resultVarSum / (double) (VECTOR_SIZE - 1);

        printf("\nThe Variance of the elements in given vector-2           : %lf", variance);

        double standardDeviation = sqrt(variance);

        printf("\nThe Standard Deviation of the elements in given vector-2 : %lf", standardDeviation);
    }
    return 0;
}
