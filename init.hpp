#ifndef INIT_HPP
#define INIT_HPP
#include <stdlib.h>
#include <time.h>
#include <algorithm>

class init{
#define DIMENSION 784

    /**Calculates the distance between two vectors in data
     * Inputs: 
     * data - flattened vector of data
     * vec1_index, vec2_index - the starting indices in data of two vectors
     */

    double distance(int *data, int *cluster, int vec1_index, int vec2_index){
        int vec1[DIMENSION], vec2[DIMENSION];
        std::copy(data+vec1_index,data+vec1_index+DIMENSION, vec1);
        std::copy(cluster+vec2_index,cluster+vec2_index+DIMENSION, vec2);
        double sum = 0;
        for(int i=0; i<DIMENSION; ++i){
            sum += (vec1[i] - vec2[i])*(vec1[i] - vec2[i]);
        }
        return sum;
    }

    int* initial_vectors(int *data, int size, int k){
        srand(time(NULL)); //seed rng
        int c[k*DIMENSION]; //initialize vector of cluster centers
        int num_samples = size/DIMENSION;
        int i, j;

        int a = rand() % size;
        int c0_index = a/DIMENSION;
        std::copy(data+c0_index, data+c0_index+DIMENSION, c);

        int current_number_clusters = 1;
        double distances[num_samples]; //keeps track of distances from all x_i to closest cluster center

        while(current_number_clusters < k){
            #pragma omp parallel shared (distances, data, current_number_clusters) private (i, j)
            {
            #pragma omp for
                for(i=0; i<num_samples; ++i){
                    double local_distances[current_number_clusters]; //keeps track of distances from current x_i to all cluster centers
                    for(j=0; j<current_number_clusters; ++j){
                        int vec1_index = DIMENSION*i; //index of start of current x_i in data
                        int vec2_index = DIMENSION*j; //index of start of current cluster center in c
                        double dist = distance(data, c, vec1_index, vec2_index);
                        local_distances[j] = dist;
                    }
                    distances[i] = *std::min_element(local_distances, local_distances+current_number_clusters);
                }
            }
            int c_index = *std::max_element(distances, distances+num_samples);
            std::copy(data+c_index, data+c_index+DIMENSION, c+current_number_clusters*DIMENSION);
            current_number_clusters++;
        }
        
        return c;
    }
};

#endif