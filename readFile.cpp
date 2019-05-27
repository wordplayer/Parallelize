#include <omp.h>
#include <math.h>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <string>

using namespace std;

int ReverseInt(int i)
{
	unsigned char ch1, ch2, ch3, ch4;
	ch1 = i & 255;
	ch2 = (i >> 8) & 255;
	ch3 = (i >> 16) & 255;
	ch4 = (i >> 24) & 255;
	return((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

void read_MNIST(string filename, double* im_arr[]) {
	ifstream file(filename.c_str(), ios::binary);

	if (file.is_open()) {
		int magic_number = 0;
		int number_of_images = 0;
		int n_rows = 0;
		int n_cols = 0;
		file.read((char*)&magic_number, sizeof(magic_number));
		magic_number = ReverseInt(magic_number);
		file.read((char*)&number_of_images, sizeof(number_of_images));
		number_of_images = ReverseInt(number_of_images);
		file.read((char*)&n_rows, sizeof(n_rows));
		n_rows = ReverseInt(n_rows);
		file.read((char*)&n_cols, sizeof(n_cols));
		n_cols = ReverseInt(n_cols);
		int i = 0, n_threads = 0, tid = 0, index = 0;

#pragma omp parallel private(i, n_threads, tid) shared(index){
		tid = omp_get_thread_num();
		if (tid == 0) {
			n_threads = omp_get_max_threads();
			cout << "Max number of threads available: " << n_threads << "\n" << endl;
		}
		cout << "Currently running thread #" << tid << "\n" << endl;
		for (; i < 10; i++)
		{
			if (index == number_of_images)
				break;
			for (int j = 0; j < n_rows; j++) {
				for (int k = 0; k < n_cols; k++) {
					unsigned char temp = 0;
					file.read((char*)&temp, sizeof(temp));
					*im_arr[index++] = (double)temp;
				}
			}
		}
		}
	}
}

int main()
{
	string filename = "t10k-images.idx3-ubyte";
	int number_of_images = 10000;
	int image_size = 28 * 28;
	ifstream file(filename.c_str(), ios::binary);
	if (!file.is_open())
	{
		system("pwd");
		exit(-1);
	}

	cout << "Going to read file" << endl;
	//read MNIST image into double vector
	double* images[number_of_images*image_size];
	read_MNIST(filename, images);
	cout << "Parallelized the loading successfully!" << endl;

	return 0;
}
