//%%cu//(only for google colab
//Author : Piyush Rajendra Chaudhari
//Roll No: BECOC311
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#define VECTOR_SIZE 10

__global__ void vectorAddition (long long *vectorA_, long long *vectorB_, long long *vectorC_) {
	vectorC_[blockIdx.x] = vectorA_[blockIdx.x] + vectorB_[blockIdx.x];
}

__global__ void vectorMatrixMultiplication (long long *vectorA_, long long *vectorB_, long long *vectorC_) {
	int row = blockIdx.y*blockDim.y+threadIdx.y;
    int col = blockIdx.x*blockDim.x+threadIdx.x;

    float tmpSum = 0;

    if (row < VECTOR_SIZE && col < VECTOR_SIZE) {
        // each thread computes one element of the block sub-matrix
        for (int indx = 0; indx < VECTOR_SIZE; indx++) {
            tmpSum += vectorA_[row * VECTOR_SIZE + indx] * vectorB_[indx * VECTOR_SIZE + col];
        }
    }
    vectorC_[row * VECTOR_SIZE + col] = tmpSum;
}

void fillVector (long long *vector_) {
	for (int indx = 0; indx < VECTOR_SIZE; indx++) {
		vector_[indx] = indx;
	}
}

void fillMatrixVector (long long *vector_) {
	for (int indx1 = 0; indx1 < VECTOR_SIZE; indx1++) {
		for (int indx2 = 0; indx2 < VECTOR_SIZE; indx2++) {
			vector_[indx1 * VECTOR_SIZE + indx2] = 1;
		}
	}
}

void printVector (long long *vector_) {
	for (int indx = 0; indx < VECTOR_SIZE; indx++) {
		printf("%lld ", vector_[indx]);
	}
}

int main(void) {
	//program for vector addition
	long long *hostVectorA, *hostVectorB, *hostVectorC;
	long long *deviceVectorA, *deviceVectorB, *deviceVectorC;
	long long memorySize = VECTOR_SIZE * sizeof(long long);
	
	// Allocate space for host vectors A, B, C and insert input values
    hostVectorA = (long long *)malloc(memorySize); 
	fillVector(hostVectorA);
    hostVectorB = (long long *)malloc(memorySize); 
	fillVector(hostVectorB);
    hostVectorC = (long long *)malloc(memorySize);
	
	// Allocate space for device vectors A, B, C
    cudaMalloc((void **)&deviceVectorA, memorySize);
    cudaMalloc((void **)&deviceVectorB, memorySize);
    cudaMalloc((void **)&deviceVectorC, memorySize);
	
	// Copy vector data from host to device
    cudaMemcpy(deviceVectorA, hostVectorA, memorySize, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceVectorB, hostVectorB, memorySize, cudaMemcpyHostToDevice);
	
	//by creating multiple blocks with single thread in it.
	dim3 blocksPerGrid(VECTOR_SIZE, 1, 1);
	dim3 threadsPerBlock(1, 1, 1);
	vectorAddition<<<blocksPerGrid, threadsPerBlock>>>(deviceVectorA, deviceVectorB, deviceVectorC);
 
    // Copy result back to host
    cudaMemcpy(hostVectorC, deviceVectorC, memorySize, cudaMemcpyDeviceToHost);
 
    printf("VECTOR A : ");
	printVector(hostVectorA);
	printf("\nVECTOR B : ");
	printVector(hostVectorB);
	printf("\nVECTOR C : ");
	printVector(hostVectorC);
    free(hostVectorA); 
	free(hostVectorB); 
	free(hostVectorC);
 
    //free gpu memory
    cudaFree(deviceVectorA); 
	cudaFree(deviceVectorB); 
	cudaFree(deviceVectorC);
	{//matrix multiplication scopes starts
	//program for matrix multiplication
	long long *hostMatrixA, *hostMatrixB, *hostMatrixC;
	long long *deviceMatrixA, *deviceMatrixB, *deviceMatrixC;
	//allocate 2-D space
	long long memoryMatrixSize = VECTOR_SIZE * VECTOR_SIZE * sizeof(long long);
	
	// Allocate space for host matrix vectors A, B, C and insert input values
    hostMatrixA = (long long *)malloc(memoryMatrixSize); 
	fillMatrixVector(hostMatrixA);
    hostMatrixB = (long long *)malloc(memoryMatrixSize); 
	fillMatrixVector(hostMatrixB);
    hostMatrixC = (long long *)malloc(memoryMatrixSize);
	
	// Allocate space for device matrix vectors A, B, C
    cudaMalloc((void **)&deviceMatrixA, memoryMatrixSize);
    cudaMalloc((void **)&deviceMatrixB, memoryMatrixSize);
    cudaMalloc((void **)&deviceMatrixC, memoryMatrixSize);
	
	// Copy vector data from host to device
    cudaMemcpy(deviceMatrixA, hostMatrixA, memoryMatrixSize, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceMatrixB, hostMatrixB, memoryMatrixSize, cudaMemcpyHostToDevice);
	
	//by creating multiple blocks with single thread in it.
	dim3 blocksPerGrid(1, 1, 1);
	dim3 threadsPerBlock(VECTOR_SIZE, VECTOR_SIZE, 1);
	if (VECTOR_SIZE * VECTOR_SIZE > 512){		
		threadsPerBlock.x = 512;
        threadsPerBlock.y = 512;
        blocksPerGrid.x = ceil(double(VECTOR_SIZE)/double(threadsPerBlock.x));
        blocksPerGrid.y = ceil(double(VECTOR_SIZE)/double(threadsPerBlock.y));
    }
	vectorMatrixMultiplication<<<blocksPerGrid, threadsPerBlock>>>(deviceMatrixA, deviceMatrixB, deviceMatrixC);
	
	// Copy result back to host
    cudaMemcpy(hostMatrixC, deviceMatrixC, memoryMatrixSize, cudaMemcpyDeviceToHost);
 
    printf("\nMATRIX VECTOR A : ");
	printVector(hostMatrixA);
	printf("\nMATRIX VECTOR B : ");
	printVector(hostMatrixB);
	printf("\nMATRIX VECTOR C : ");
	printVector(hostMatrixC);
    free(hostMatrixA); 
	free(hostMatrixB); 
	free(hostMatrixC);
 
    //free gpu memory
    cudaFree(deviceMatrixA); 
	cudaFree(deviceMatrixB); 
	cudaFree(deviceMatrixC);
	}//matrix multiplication scopes ends
	return 0;
}
