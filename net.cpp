#include "net.hpp"
#include <assert.h>

void Net::setup(float learning_rate, int maxIteration, int batch_size) {
	this->learning_rate = learning_rate;
	this->maxIteration = maxIteration;
	this->batch_size = batch_size;
}

void Net::setDataLayer(Layer* layer) {
	if (!allLayers.empty()) {

	}
	allLayers.push_back(layer);
}

void Net::addLayer(Layer* layer) {
	Layer* lastLayer = allLayers[allLayers.size() - 1];
	int num_input = lastLayer->getNumOutput();
	layer->setup(num_input, learning_rate);
	allLayers.push_back(layer);
}

void Net::setLoss(Loss* loss) {
	Layer* lastLayer = allLayers[allLayers.size() - 1];
	int num_input = lastLayer->getNumOutput();
	loss->setup(num_input, learning_rate);
	this->loss = loss;
}

void Net::train() {
	for (int i = 0; i < maxIteration; i++) {
		Mat inputs = allLayers[0]->getInputData();
		Mat labels = allLayers[0]->getLabels();

		vector<Mat> vecInputs;
		vector<Mat> vecLabels;
		randomData(inputs, labels, vecInputs, vecLabels);

		for (int k = 0; k < vecInputs.size(); k++) {
			Mat input = vecInputs[k];
			Mat label = vecLabels[k];

			vector<Mat> outputs;
			outputs.push_back(input);
			for (int j = 1; j < allLayers.size(); j++) {
				Mat output;
				allLayers[j]->forward(outputs[j - 1], output);
				outputs.push_back(output);
			}

			int num_layers = outputs.size();
			assert(outputs[num_layers - 1].cols == label.cols);
			
			Mat errors = Mat::zeros(1, labels.cols, CV_32FC1);

			for (int j = 0; j < label.cols; j++) {
				float loss = outputs[num_layers - 1].at<float>(0, j) - label.at<float>(0, j);
				errors.at<float>(0, j) = loss;
				if (i % 1000 == 0)
					printf("iteration %i, loss: %f\n", i, loss);

			}
			vector<Mat> residual_error(allLayers.size());

			/*Mat errors;
			loss->compute(outputs[outputs.size() - 1], label, lossErrors);
			loss->compute(outputs[outputs.size() - 1], label, errors);*/

			
			residual_error[allLayers.size() - 1] = errors;
			for (int j = allLayers.size() - 1; j > 0; j--) {
				Mat error = allLayers[j]->backward(outputs[j - 1], outputs[j], residual_error[j]);
				residual_error[j - 1] = error;
			}
			
		}
		for (int k = allLayers.size() - 1; k > 0; k--) {
			allLayers[k]->updateWeight(vecInputs.size());
		}
	}
}

Mat Net::predict(Mat input) {
	vector<Mat> outputs;
	outputs.push_back(input);
	for (int j = 1; j < allLayers.size(); j++) {
		Mat output;
		Mat w = allLayers[j]->getW();
		allLayers[j]->forward(outputs[j - 1], output);
		outputs.push_back(output);
	}
	return outputs[outputs.size() - 1];
}
void Net::randomData(Mat inputs, Mat labels, vector<Mat> &vecInputs, vector<Mat> &vecLabels) {
	vector<int> index;
	for (int i = 0; i < inputs.rows; i++) {
		index.push_back(i);
	}
	random_shuffle(index.begin(), index.end());

	for (int i = 0; i < index.size() && i < batch_size; i++) {
		Mat tempInput = inputs.rowRange(index[i], index[i] + 1);
		Mat tempLabel = labels.rowRange(index[i], index[i] + 1);
		vecInputs.push_back(tempInput);
		vecLabels.push_back(tempLabel);
	}
}