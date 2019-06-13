#pragma once
#include "device_launch_parameters.h"
#include "k_means_4dim.h"

__device__ double calculatedistanceGPU(unit* point1, unit* point2);
__global__ void closestcentroidGPU(unit* points, unit* centroids, int numofcentr, int numofpoints);
__global__ void closestcentroidSharedGPU(unit* points, unit* centroids, int numofcentr, int numofpoints);
