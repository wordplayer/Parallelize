#include "init.hpp"
#include "readMNIST.cpp"
#include <stdlib.h>
#include <iostream>


void usage(char* program_name){
    cerr << program_name << " called with incorrect arguments." << endl;
    cerr << "Usage: " << program_name
        << " data_file num_clusters" << endl;
    exit(-1);
}

int main(int argc, char** argv)
{
    char* filename;
    int* data;
    int* labels;
    int* initial_clusters;
    int k = 10;
    
    if(argc==1 || argc>3) {usage(argv[0]);}
    if(argc==2) {filename = argv[1];}
    if(argc==3) {
        filename = argv[1];
        k = atoi(argv[2]);
    }

    read_Mnist(filename, data);
    read_Mnist_Label(filename, labels);
    initial_clusters = initial_vectors(data, 10000*784, k);

    return 0;
}