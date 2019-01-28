#include <random>
#include <math.h>
#include <algorithm>
#include <assert.h>

#include "layer.hpp"

DataLayer::DataLayer(Mat inputData, Mat labels) {
	assert(inputData.rows == labels.rows);
	this->inputData = inputData;
	this->labels = labels;
}

void DataLayer::setup(int num_input, float learning_rate) {

}

void DataLayer::forward(Mat &input, Mat &output) {

}

Mat DataLayer::backward(Mat input, Mat output, Mat error) {
	return Mat();
}

Mat DataLayer::getInputData() {
	return inputData;
}

Mat DataLayer::getLabels() {
	return labels;
}

int DataLayer::getNumOutput() {
	return inputData.cols;
}

void DataLayer::updateWeight(int records) {
}

Mat DataLayer::getW() {
	return Mat();
}

InnerProductLayer::InnerProductLayer(int num_output) {
	this->num_output = num_output;
}

void InnerProductLayer::setup(int num_input, float learning_rate) {
	this->learning_rate = learning_rate;
	w = Mat(num_input, num_output, CV_32FC1, Scalar(0));
	b = Mat(1, num_output, CV_32FC1, Scalar(1));
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

void InnerProductLayer::forward(Mat &input, Mat &output) {
	//resize(input, input, Size(input.size(), 1));
	Mat result = input * w;
	for (int i = 0; i < result.cols; i++) {
		float num = result.at<float>(0, i);
		result.at<float>(0, i) = 1 / (1 + exp(-num));
	}
	output = result;
}

Mat InnerProductLayer::backward(Mat input, Mat output, Mat error) {
	Mat this_layer_error(1, input.cols, CV_32FC1, Scalar(0));
	float error_sum = 0;
	assert(error.cols == w.cols);

	//Mat dSigmoid = output.mul(1 - output);
	//Mat weight_delta = input.t() * (dSigmoid.mul(error));
	//weight_deltas += weight_delta;

	for (int i = 0; i < error.cols; i++)
		error_sum += error.at<float>(0, i);

	//this_layer_error = (dSigmoid.mul(error)) * w.t();

	for (int i = 0; i < w.rows; i++) {
		float temp = 0;
		for (int j = 0; j < w.cols; j++) {
			float dSigmoid = output.at<float>(0, j) * (1 - output.at<float>(0, j));
				
			float weight_delta = error.at<float>(0, j) * dSigmoid * input.at<float>(0, i);
			//update weight
			weight_deltas.at<float>(i, j) += weight_delta;
			
			//w.at<float>(i, j) = w.at<float>(i, j) - result * learning_rate;

			temp += w.at<float>(i, j) * dSigmoid * error.at<float>(0, j);
		}
		this_layer_error.at<float>(0, i) = temp;
	}


	//w.rows = num_input, w.cols = num_output
	//for (int i = 0; i < w.rows; i++) {
	//	float temp = 0;
	//	for (int j = 0; j < w.cols; j++) {
	//		float dSigmoid = output[j] * (1 - output[j]);
	//			
	//		float weight_delta = error[j] * dSigmoid * input[i];
	//		//update weight
	//		weight_deltas.at<float>(i, j) += weight_delta;
	//		
	//		//w.at<float>(i, j) = w.at<float>(i, j) - result * learning_rate;

	//		temp += w.at<float>(i, j) * dSigmoid * error[j];
	//	}
	//	this_layer_error.push_back(temp);
	//}
	return this_layer_error;
}

Mat InnerProductLayer::getInputData() {
	return Mat();
}

Mat InnerProductLayer::getLabels() {
	return Mat();
}

int InnerProductLayer::getNumOutput() {
	return num_output;
}

void InnerProductLayer::updateWeight(int records) {
	for (int i = 0; i < w.rows; i++) {
		for (int j = 0; j < w.cols; j++) {
			w.at<float>(i, j) = w.at<float>(i, j) - (learning_rate * weight_deltas.at<float>(i, j) / records);
		}
	}
	weight_deltas.setTo(0);
}

Mat InnerProductLayer::getW() {
	return w;
}


L2LossLayer::L2LossLayer(int num_output) {
	this->num_output = num_output;
}

void L2LossLayer::setup(int num_input, float learning_rate) {
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

void L2LossLayer::forward(Mat &input, Mat &output) {
	//resize(input, input, Size(input.size(), 1));
	Mat result = input * w;
	output = result;


}

Mat L2LossLayer::backward(Mat input, Mat output, Mat error) {
	Mat this_layer_error(1, input.cols, CV_32FC1, Scalar(0));
	float error_sum = 0;
	assert(error.cols == w.cols);

	Mat weight_delta = error * input;

	weight_deltas += weight_delta.t();

	Mat temp = error * w.t();

	this_layer_error += temp;

	//for (int i = 0; i < w.rows; i++) {
	//	float temp = 0;
	//	for (int j = 0; j < w.cols; j++) {
	//		//float dSigmoid = output.at<float>(0, j) * (1 - output.at<float>(0, j));

	//		float weight_delta = error.at<float>(0, j) * input.at<float>(0, i);
	//		//update weight
	//		weight_deltas.at<float>(i, j) += weight_delta;

	//		//w.at<float>(i, j) = w.at<float>(i, j) - result * learning_rate;

	//		temp += w.at<float>(i, j) * error.at<float>(0, j);
	//	}
	//	this_layer_error.at<float>(0, i) = temp;
	//}
	return this_layer_error;
}

Mat L2LossLayer::getInputData() {
	return Mat();
}

Mat L2LossLayer::getLabels() {
	return Mat();
}

int L2LossLayer::getNumOutput() {
	return num_output;
}

void L2LossLayer::updateWeight(int records) {
	for (int i = 0; i < w.rows; i++) {
		for (int j = 0; j < w.cols; j++) {
			w.at<float>(i, j) = w.at<float>(i, j) - (learning_rate * weight_deltas.at<float>(i, j) / records);
		}
	}
	weight_deltas.setTo(0);
}

Mat L2LossLayer::getW() {
	return w;
}