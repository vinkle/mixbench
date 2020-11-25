/**
 * main-cuda.cpp: This file is part of the mixbench GPU micro-benchmark suite.
 *
 * Contact: Elias Konstantinidis <ekondis@gmail.com>
 **/

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <unistd.h> 
#include <pthread.h> 
#include <cuda_runtime.h>
#include <string.h>
#include "lcutil.h"
#include "mix_kernels_cuda.h"
#include "version_info.h"

#ifdef READONLY
#define VECTOR_SIZE (32*1024*1024)
#else
#define VECTOR_SIZE (256*1024*1024)
#endif
#define NUM_ITERATIONS 1000

void *gputhread0(void *vargp) 
{
	cudaSetDevice(0);
	for(int i = 0; i < NUM_ITERATIONS; ++i)
	{
		double *c;
		c = (double*)malloc(datasize);
		mixbenchGPU(c, VECTOR_SIZE);
		free(c);
	}
} 

void *gputhread1(void *vargp) 
{
	cudaSetDevice(1);
	for(int i = 0; i < NUM_ITERATIONS; ++i)
	{
		double *c;
		c = (double*)malloc(datasize);
		mixbenchGPU(c, VECTOR_SIZE);
		free(c);
	}
} 


void *gputhread2(void *vargp) 
{
	cudaSetDevice(2);
	for(int i = 0; i < NUM_ITERATIONS; ++i)
	{
		double *c;
		c = (double*)malloc(datasize);
		mixbenchGPU(c, VECTOR_SIZE);
		free(c);
	}
} 


void *gputhread3(void *vargp) 
{
	cudaSetDevice(3);
	for(int i = 0; i < NUM_ITERATIONS; ++i)
	{
		double *c;
		c = (double*)malloc(datasize);
		mixbenchGPU(c, VECTOR_SIZE);
		free(c);
	}
} 



int main(int argc, char* argv[]) {
#ifdef READONLY
	printf("mixbench/read-only (%s)\n", VERSION_INFO);
#else
	printf("mixbench/alternating (%s)\n", VERSION_INFO);
#endif

	unsigned int datasize = VECTOR_SIZE*sizeof(double);

	cudaSetDevice(0);
	StoreDeviceInfo(stdout);

	size_t freeCUDAMem, totalCUDAMem;
	cudaMemGetInfo(&freeCUDAMem, &totalCUDAMem);
	printf("Total GPU memory %lu, free %lu\n", totalCUDAMem, freeCUDAMem);
	printf("Buffer size:          %dMB\n", datasize/(1024*1024));
	
	pthread_t thread_id0, thread_id1, thread_id2, thread_id3;
	pthread_create(&thread_id0, NULL, gputhread0, NULL);
	pthread_create(&thread_id1, NULL, gputhread1, NULL);
	pthread_create(&thread_id2, NULL, gputhread2, NULL);
	pthread_create(&thread_id3, NULL, gputhread3, NULL);
	pthread_join(thread_id0, NULL); 
	pthread_join(thread_id1, NULL); 
	pthread_join(thread_id2, NULL); 
	pthread_join(thread_id3, NULL); 
	
	printf("Finishing processing Thread\n"); 
	

	return 0;
}
