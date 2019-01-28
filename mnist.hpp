#include <iostream>
#include <string>
#include <fstream>
#include <ctime>
#include <cv.h>
#include <highgui.h>

using namespace cv;
using namespace std;

//小端存储转换
int reverseInt(int i);

//读取image数据集信息
Mat read_mnist_image(const string fileName);

//读取label数据集信息
Mat read_mnist_label(const string fileName);