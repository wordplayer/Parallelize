#include "init.hpp"
#include "readMNIST.cpp"
#include <stdlib.h>
#include <iostream>

#define TRAINING_SIZE 10,000
#define DIMENSION 784

_global_ void kmeans(int *data, int *initial_clusters){
    //TODO ZENAAAAAAAAAAAA this one's your part
}

void usage(char* program_name){
    cerr << program_name << " called with incorrect arguments." << endl;
    cerr << "Usage: " << program_name
        << " data_file num_clusters" << endl;
    exit(-1);
}

int main(int argc, char** argv)
{
    char *filename;
    int h_data, h_labels, h_initial_clusters;
    int *d_data, *d_labels, *d_initial_clusters, *d_distances, *d_clusters;
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
    read_Mnist_Label(filename, &h_labels);
    h_initial_clusters = *initial_vectors(&h_data, 10000*784, k);

    //Allocate global memory on device
    cudaMalloc((void **)&d_data, int_size*TRAINING_SIZE*DIMENSION);
    cudaMalloc((void **)&d_labels, int_size*TRAINING_SIZE);
    cudaMalloc((void **)&d_initial_clusters, int_size*k*DIMENSION);
    cudaMalloc((void **)&d_distances, double_size*TRAINING_SIZE*k);
    //TODO I think we will have to use a 2D array to keep track of clusters

    //Copy host values to device variables
    cudaMemcpy(d_data, &h_data, int_size*TRAINING_SIZE*DIMENSION, cudaMemcpyHostToDevice);
    cudaMemcpy(d_labels, &h_labels, int_size*TRAINING_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(d_initial_clusters, &h_initial_clusters, int_size*k*DIMENSION, cudaMemcpyHostToDevice);

    //TODO kernel function goes here

    //TODO We need to copy the cluster vector (or 2D matrix?) back to the host
    cudaMemcpy()

    //Deallocate memory
    cudaFree(d_data);
    cudaFree(d_labels);
    cudaFree(d_initial_clusters);
    cudaFree(d_distances);

    return 0;
}