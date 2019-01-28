#include <random>
#include <math.h>
#include <assert.h>
#include <math.h>
#include <vector>
#include "loss.hpp"

L2Loss::L2Loss(int num_output) {
	this->num_output = num_output;
}

void L2Loss::setup(int num_input, float learning_rate) {
	this->learning_rate = learning_rate;
	w = Mat(num_input, num_output, CV_32FC1, Scalar(0));
	b = Mat(1, num_output, CV_32FC1);
	weight_deltas = Mat(num_input, num_output, CV_32FC1, Scalar(0));
	default_random_engine e;
	normal_distribution<float> n(0, pow(num_input, -0.5));
	for (int i = 0; i < w.rows; i++) {
		for (int j = 0; j < w.cols; j++) {
			w.at<float>(i, j) = n(e);
		}
	}
	//for (int i = 0; i < b.cols; i++) {
	//	b.at<float>(0, i) = n(e);
	//}
}

void L2Loss::compute(Mat input, Mat label, Mat &lossError) {
	assert(input.rows == labels.rows);
	Mat output = input * w;
	Mat errors = Mat::zeros(1, label.cols, CV_32FC1);

	Mat left, right;
	Mat output1;
	Mat output2;
	cv::log(output, output1);
	cv::log(1 - output, output2);
	multiply(label, output1, left);
	multiply((1 - label), output2, right);

	Mat lossMat = -(left + right);
	float loss = sum(lossMat)[0];

	//printf("loss: %f\n", loss);
	for (int j = 0; j < input.cols; j++) {
		float loss = output.at<float>(0, j) - label.at<float>(0, j);
		errors.at<float>(0, j) = loss;
	}

	lossError = Mat(1, input.cols, CV_32FC1, Scalar(0));
	float error_sum = 0;
	assert(error.cols == w.cols);

	Mat weight_delta = errors * input;

	weight_deltas += weight_delta.t();

	Mat temp = errors * w.t();

	lossError += temp;

}

void L2Loss::forward(Mat &input, Mat &output) {
	output = input * w;
}


SoftmaxLoss::SoftmaxLoss() {
}

void SoftmaxLoss::setup(int num_input, float learning_rate) {

}

void SoftmaxLoss::compute(Mat input, Mat label, Mat &lossError) {
	Mat output;
	forward(input, output);
	assert(output.cols == label.cols);
	/*Point outputMaxLoc;
	Point labelMaxLoc;
	minMaxLoc(output, 0, 0, 0, &outputMaxLoc);
	minMaxLoc(label, 0, 0, 0, &labelMaxLoc);
	for (int i = 0; i < label.cols; i++) {

	}*/
	lossError = output - label;
}

void SoftmaxLoss::forward(Mat &input, Mat &output) {
	vector<float> nums;
	float sum = 0;
	for (int i = 0; i < input.cols; i++) {
		float num = exp(input.at<float>(0, i));
		sum += num;
		nums.push_back(num);
	}
	for (int i = 0; i < input.cols; i++)
		output.at<float>(0, i) = nums[i] / sum;
}