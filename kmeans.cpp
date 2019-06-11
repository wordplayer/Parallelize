#include "init.hpp"
#include "readMNIST.cpp"
#include <stdlib.h>
#include <iostream>
#include <limits>

#define TRAINING_SIZE 9984
#define DIMENSION 784
#define INFINITY std::numeric_limits<double>::max()
#define NUM_BLOCKS 39
#define NUM_THREADS 256 //recommended best number of threads according to CUDA manual
#define k 10



__global__ void kmeans(double *data, int *initial_clusters){

    __shared__ double temp[k*DIMENSION]; //shared cluster center data array
    __shared__ double sum[k];
    __shared__ double counts[k];
    __shared__ double distances[NUM_THREADS];
    __shared__ double distances_min[NUM_THREADS];

    int tid = threadIdx.x + blockDim.x*blockIdx.x; //index of a thread in the grid

    //Initialize shared memory
    for(int i=threadIdx.x; i<k*DIMENSION; i += blockDim.x)
        temp[i] = iniitial_clusters[i];

    if(threadIdx.x < k){
        sum[threadIdx.x] = 0;
        counts[threadIdx.x] = 0;
    }

    distances[threadIdx.x] = 0;
    distances_min[threadIdx.x] = INFINITY;

    __syncthreads();



    for(int iter=0; iter<1000; ++iter){
        double curr_element;
        for(int j=0; j<k; j++){
            for(int i=tid*DIMENSION; i<DIMENSION; ++i){
                distance[tid] += (data[i]-temp[j+i])*(data[i]-temp[j+i]);
            }
        }
        
    }


    while(tid < TRAINING_SIZE){ 
        double dist = 0;
        double curr_vector[DIMENSION];
        for(int i=0; i<DIMENSION; ++i){
            curr_vector[i] = data[tid*DIMENSION+i];
        }
        
        // Classify the data point into a cluster
        for(int i=0; i<k; ++i){
            for(int j=0; j<DIMENSION; ++j){
                dist += (curr_vector[j] - temp[i*DIMENSION+j])
                *(curr_vector[j] - temp[i*DIMENSION+j]);
            }
            if(dist<min_dist){
                min_dist = dist;
                clusters[tid] = i;
                distances[tid] = min_dist;
            }
        }
        
        // Update means
        int cluster = clusters[tid];
        numEachCluster[cluster]++;
        int size = numEachCluster[cluster];
        double mean[DIMENSION];
        for(int i=0; i<DIMENSION; i++)
            mean[i] = means[i+DIMENSION*cluster];
        double * meanPtr = mean;
        double * meanUpdate = updateMean(meanPtr, size, data);
        
        for(int i=DIMENSION*cluster; i<DIMENSION*(cluster+1); i++)
        {
            temp[i] = meanUpdate[i];
        }
        
        tid += stride;
        
    }
}


double * updateMean(double * mean, int size, double * data)
{
    for(int i=0; i<DIMENSION; i++)
    {
        double m = mean[i];
        m = (double)(m*(size-1) + data[i]) / (double)size;
        mean[i] = m;
    }
    return mean;
}



void usage(char* program_name){
    cerr << program_name << " called with incorrect arguments." << endl;
    cerr << "Usage: " << program_name
        << " data_filename num_clusters" << endl;
    exit(-1);
}

int main(int argc, char** argv)
{
    char *filename;
    int h_data, h_labels, h_initial_clusters;
    //d_clusters is an N x N array
    int *d_labels, *d_initial_clusters, *d_clusters;
    double *d_distance, *d_data;
    int int_size = sizeof(int);
    int double_size = sizeof(double);
    
    if(argc==1 || argc>3) {usage(argv[0]);}
    if(argc==2) {filename = argv[1];}
    if(argc==3) {
        filename = argv[1];
        k = atoi(argv[2]);
    }

    read_Mnist(filename, &h_data); //TODO BOOOOSE you'll have to make it so that we can input an array instead of vector
    read_Mnist_Label(filename, &h_labels); //TODO same with this one
    h_initial_clusters = *initial_vectors(&h_data, 10000*784, k);

    //Allocate global memory on device
    cudaMalloc((void **)&d_data, double_size*TRAINING_SIZE*DIMENSION); //1D falttened array of data (global memoryh)
    cudaMalloc((void **)&d_initial_clusters, int_size*k*DIMENSION); //1D array to keep track of cluster centers
    cudaMalloc((void **)&d_clusters, int_size*TRAINING_SIZE); //1D array of cluster assignments for each point
    cudaMalloc((void **)&d_sum, double_size*k); //Keep track of sums for calculating means
    cudaMalloc((void **)&d_counts, double_size*k); //Keep track of counts of data points in each cluster

    //Copy host values to device variables
    cudaMemcpy(d_data, &h_data, double_size*TRAINING_SIZE*DIMENSION, cudaMemcpyHostToDevice);
    cudaMemcpy(d_initial_clusters, &h_initial_clusters, int_size*k*DIMENSION, cudaMemcpyHostToDevice);

    //TODO kernel function goes here
    kmeans<<d_data, d_initial_clusters>>

    //TODO We need to copy the cluster vector (or 2D matrix?) back to the host
    cudaMemcpy()

    //Deallocate memory
    cudaFree(d_data);
    cudaFree(d_labels);
    cudaFree(d_initial_clusters);
    cudaFree(d_clusters);
    cudaFree(d_distances);

    return 0;
}