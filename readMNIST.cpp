#include <math.h>
#include <iostream>
#include <vector>
#include <fstream>

using namespace std;

int ReverseInt (int i)
{
    unsigned char ch1, ch2, ch3, ch4;
    ch1 = i & 255;
    ch2 = (i >> 8) & 255;
    ch3 = (i >> 16) & 255;
    ch4 = (i >> 24) & 255;
    return((int) ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

void read_Mnist(string filename, vector<vector<double> > &vec)
{
    ifstream file (filename, ios::binary);

    if (file.is_open())
    {
        int magic_number = 0;
        int number_of_images = 0;
        int n_rows = 0;
        int n_cols = 0;
        file.read((char*) &magic_number, sizeof(magic_number));
        magic_number = ReverseInt(magic_number);
        file.read((char*) &number_of_images,sizeof(number_of_images));
        number_of_images = ReverseInt(number_of_images);
        file.read((char*) &n_rows, sizeof(n_rows));
        n_rows = ReverseInt(n_rows);
        file.read((char*) &n_cols, sizeof(n_cols));
        n_cols = ReverseInt(n_cols);
        for(int i = 0; i < number_of_images; ++i)
        {
            vector<double> tp;
            for(int r = 0; r < n_rows; ++r)
            {
                for(int c = 0; c < n_cols; ++c)
                {
                    unsigned char temp = 0;
                    file.read((char*) &temp, sizeof(temp));
                    tp.push_back((double)temp);
                }
            }
            vec.push_back(tp);
        }
    }
}

void read_Mnist_Label(string filename, vector<double> &vec)
{
    ifstream file (filename, ios::binary);
    if (file.is_open())
    {
        int magic_number = 0;
        int number_of_images = 0;
        int n_rows = 0;
        int n_cols = 0;
        file.read((char*) &magic_number, sizeof(magic_number));
        magic_number = ReverseInt(magic_number);
        file.read((char*) &number_of_images,sizeof(number_of_images));
        number_of_images = ReverseInt(number_of_images);
        for(int i = 0; i < number_of_images; ++i)
        {
            unsigned char temp = 0;
            file.read((char*) &temp, sizeof(temp));
            vec[i]= (double)temp;
        }
    }
}

int main()
{
    string filename = "t10k-images.idx3-ubyte";
    int number_of_images = 10000;
    int image_size = 28 * 28;
    ifstream file(filename, ios::binary);
    if(!file.is_open())
    {
        system("pwd");
        exit(-1);
    }


    //read MNIST iamge into double vector
    vector<vector<double> > vec;
    read_Mnist(filename, vec);
    cout<<vec.size()<<endl;
    cout<<vec[0].size()<<endl;

    //read MNIST label into double vector
    vector<double> vec2(number_of_images);
    read_Mnist_Label(filename, vec2);
    cout<<vec.size()<<endl;


    return 0;
}