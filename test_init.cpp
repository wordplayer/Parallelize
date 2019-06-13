#include "init.hpp"
#include <cstdlib>

int main() {
	double* vec[] = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0 };
	int size = 10;
	int k = 3;
	double* cluster[] = initial_vectors(vec, size, k);
	for (int i = 0; i < k; i++)
		cout << *cluster[i] << endl;
}