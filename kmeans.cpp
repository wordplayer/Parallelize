#include "init.hpp"
#include "readMNIST.cpp"
#include <stdlib.h>
#include <iostream>
#include <limits>

#define TRAINING_SIZE 10,000
#define DIMENSION 784
#define INFINITY std::numeric_limits<double>::max()


__global__ void kmeans(double *data, int *initial_clusters, 
    int *clusters, double *distances, int k){

    __shared__ int temp[BLOCK_SIZE]; //shared initial cluster array

    int gindex = threadIdx.x + blockDim.x*blockIdx.x;
    int stride = blockDim.x*gridDim.x;
    int tid = threadId.x;
    if(tid < k){
        temp[tid] = initial_clusters[tid];
    }

    __syncthreads();

    double min_dist = INFINITY;
    while(tid < TRAINING_SIZE){
        double dist = 0;
        double curr_vector[DIMENSION];
        for(int i=0; i<DIMENSION; ++i){
            curr_vector[i] = data[tid*DIMENSION+i];
        }
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
        tid += stride;
    }

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
    double *d_distances, *d_data;
    int k = 10;
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
    cudaMalloc((void **)&d_data, double_size*TRAINING_SIZE*DIMENSION); //1D falttened array of data
    cudaMalloc((void **)&d_labels, int_size*TRAINING_SIZE); //1D array of data labels
    cudaMalloc((void **)&d_initial_clusters, int_size*k*DIMENSION); //1D array to keep track of cluster centers
    cudaMalloc((void **)&d_clusters, int_size*TRAINING_SIZE); //1D array of cluster assignments for each point
    cudaMalloc((void **)&d_distances, double_size*k); //1D array keeps track of sums of distances of each point in cluster to cluster center
    //TODO I think we will have to use a 2D array to keep track of clusters

    //Copy host values to device variables
    cudaMemcpy(d_data, &h_data, double_size*TRAINING_SIZE*DIMENSION, cudaMemcpyHostToDevice);
    cudaMemcpy(d_labels, &h_labels, int_size*TRAINING_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(d_initial_clusters, &h_initial_clusters, int_size*k*DIMENSION, cudaMemcpyHostToDevice);

    //TODO kernel function goes here

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