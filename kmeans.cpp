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



__global__ void kmeans(double *data, int *initial_clusters, int *d_sum, int *d_counts){

    __shared__ double temp[k*DIMENSION]; //shared cluster center data array
    //zena __shared__ double sum[k];
    //zena __shared__ double counts[k];
    //zena __shared__ double distances[NUM_THREADS];
    //zena __shared__ double distances_min[NUM_THREADS];

    int tid = threadIdx.x + blockDim.x*blockIdx.x; //index of a thread in the grid

    //Initialize shared memory
    for(int i=threadIdx.x; i<k*DIMENSION; i += blockDim.x)
        temp[i] = iniitial_clusters[i];

    /* zena
    if(threadIdx.x < k){
        sum[threadIdx.x] = 0;
        counts[threadIdx.x] = 0;
    }*/
    
    //zena distances[threadIdx.x] = 0;
    //zena distances_min[threadIdx.x] = INFINITY;

    __syncthreads();
    
    if(tid < TRAINING_SIZE){
        for(int iter=0; iter<1000; ++iter)
        {
            double min_dist = INFINITY;
            /*Initialize a vector for this thread that wills store the data point that this thread
            is calculating a distance for. (aditya)*/
            double data_vector[DIMENSION];
            for(int i=0; i<DIMENSION; ++i)
            {
                data_vector[i] = data[(DIMENSION*tid)+i];
            }

            for(int j=0; j<k; ++j)
            {
                /*Calculate distance from data point to centroid j one coordinate at a time. */
                double distance = 0;
                for(int i=tid*DIMENSION; i<DIMENSION; ++i)
                {
                    distance += (data_vector[i]-temp[j*DIMENSION+i])*(data_vector[i]-temp[j*DIMENSION+i]);
                }

                /*If distance from data point to centroid j is less than current min_dist,
                update min_dist and assign data point to centroid j in d_clusters (global memory). */
                if(distance < min_dist)
                {
                    min_dist = distance;
                    d_clusters[tid] = j;
                }
            }
            
            int assigned_cluster = d_clusters[tid];
            //d_counts[assigned_cluster]++;
            atomicAdd(&(d_counts[assigned_cluster]), 1);

            for(int s=0; s<DIMENSION; s++)
            {
                //d_sum[s+assigned_cluster*DIMENSION] += data[tid+s];
                atomicAdd(&(d_sum[s+assigned_cluster*DIMENSION]), data[tid+s]);
            }
            __syncthreads();

            /*Reassign cluster centers */
            if(tid < k*DIMENSION)
            {
                //tid%DIMENSION gives the correct k to divide by
                temp[tid] = d_sum[tid] / d_counts[(tid%DIMENSION)];
            }

            /*Reinitialize d_sum and d_counts */
            if(tid < k*DIMENSION)
            {
                d_sum[tid] = 0;
                if(tid < k) d_counts[tid] = 0;
            }
            __syncthreads();
        }
    }

    /*Copy cluster centers from temp to initial_clusters */
    if(tid < k*DIMENSION)
    {
        initial_clusters[tid] = temp[tid];
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
    //char *filename;
    int h_labels, h_counts[k];
    double *h_initial_clusters;
    double h_data, h_sum[k*DIMENSION];

    int *d_counts;
    double *d_data, *d_sum, *d_initial_clusters;
    int int_size = sizeof(int);
    int double_size = sizeof(double);
    
    /*
    if(argc==1 || argc>3) {usage(argv[0]);}
    if(argc==2) {filename = argv[1];}
    if(argc==3) {
        filename = argv[1];
        k = atoi(argv[2]);
    }*/

    read_Mnist(filename, &h_data); //TODO BOOOOSE you'll have to make it so that we can input an array instead of vector
    read_Mnist_Label(filename, &h_labels); //TODO same with this one
    h_initial_clusters = initial_vectors(&h_data, 10000*784, k);

    for(int i=0; i<k*DIMENSION; ++i)
    {
        h_sum[i] = 0;
    } 
    for(int i=0; i<k; ++i)
    {
        h_counts[i] = 0;
    }

    //Allocate global memory on device
    cudaMalloc((void **)&d_data, double_size*TRAINING_SIZE*DIMENSION); //1D falttened array of data (global memoryh)
    cudaMalloc((void **)&d_initial_clusters, double_size*k*DIMENSION); //1D array to keep track of cluster centers
    cudaMalloc((void **)&d_sum, double_size*k*DIMENSION); //Keep track of sums for calculating means
    cudaMalloc((void **)&d_counts, int_size*k); //Keep track of counts of data points in each cluster

    //Copy host values to device variables
    cudaMemcpy(d_data, &h_data, double_size*TRAINING_SIZE*DIMENSION, cudaMemcpyHostToDevice);
    cudaMemcpy(d_initial_clusters, h_initial_clusters, double_size*k*DIMENSION, cudaMemcpyHostToDevice);
    cudaMemcpy(d_sum, &h_sum, double_size*k*DIMENSION, cudaMemcpyHostToDevice);
    cudaMemcpy(d_counts, &h_counts, int_size*k, cudaMemcpyHostToDevice);


    //TODO kernel function goes here
    kmeans<<d_data, d_initial_clusters, d_sum, d_counts>>

    //TODO We need to copy the cluster vector back to the host
    cudaMemcpy(h_initial_clusters, d_initial_clusters, int_size*k*DIMENSION, cudaMemcpyDeviceToHost);

    //Deallocate memory
    cudaFree(d_data);
    cudaFree(d_initial_clusters);
    cudaFree(d_sum);
    cudaFree(d_counts);

    return 0;
}