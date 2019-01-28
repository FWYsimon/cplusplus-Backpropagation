#include <vector>
#include <cv.h>
#include <highgui.h>

using namespace cv;
using namespace std;

class Loss{
public:
	virtual void setup(int num_input, float learning_rate) = 0;
	virtual void compute(Mat input, Mat label, Mat &lossError) = 0;
	virtual void forward(Mat &input, Mat &output) = 0;
};

class L2Loss :public Loss{
public:
	L2Loss(int num_output);
	void setup(int num_input, float learning_rate);
	void compute(Mat input, Mat label, Mat &lossError);
	void forward(Mat &input, Mat &output);
private:
	float learning_rate;
	int num_output;
	Mat weight_deltas;
	Mat w;
	Mat b;
};

class SoftmaxLoss :public Loss{
public:
	SoftmaxLoss();
	void setup(int num_input, float learning_rate);
	void compute(Mat input, Mat label, Mat &lossError);
	void forward(Mat &input, Mat &output);
};