#pragma  once

#include <vector>
#include <cv.h>
#include <highgui.h>

using namespace cv;
using namespace std;

class Layer {
public:
	virtual void setup(int num_input, float learning_rate) = 0;
	virtual void forward(Mat &input, Mat &output) = 0;
	virtual Mat backward(Mat input, Mat output, Mat error) = 0;
	virtual int getNumOutput() = 0;
	virtual Mat getInputData() = 0;
	virtual Mat getLabels() = 0;
	virtual void updateWeight(int records) = 0;
	virtual Mat getW() = 0;
};

class DataLayer : public Layer{
public:
	DataLayer(Mat inputData, Mat labels);
	void setup(int num_input, float learning_rate);
	void forward(Mat &input, Mat &output);
	Mat backward(Mat input, Mat output, Mat error);
	Mat getInputData();
	Mat getLabels();
	int getNumOutput();
	void updateWeight(int records);
	Mat getW();
private:
	Mat inputData;
	Mat labels;
};

class InnerProductLayer : public Layer{
public:
	InnerProductLayer(int num_output);
	void setup(int num_input, float learning_rate);
	void forward(Mat &input, Mat &output);
	Mat backward(Mat input, Mat output, Mat error);
	Mat getInputData();
	Mat getLabels();
	int getNumOutput();
	void updateWeight(int records);
	Mat getW();
private:
	float learning_rate;
	int num_output;
	Mat weight_deltas;
	Mat w;
	Mat b;
};

class L2LossLayer : public Layer{
public:
	L2LossLayer(int num_output);
	void setup(int num_input, float learning_rate);
	void forward(Mat &input, Mat &output);
	Mat backward(Mat input, Mat output, Mat error);
	Mat getInputData();
	Mat getLabels();
	int getNumOutput();
	void updateWeight(int records);
	Mat getW();
private:
	float learning_rate;
	int num_output;
	Mat weight_deltas;
	Mat w;
	Mat b;
};