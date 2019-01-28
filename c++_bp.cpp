#include <iostream>
#include "net.hpp"

using namespace std;

int main() {
	Net net;
	net.setup(0.5f, 100000, 8);

	Mat inputs;
	Mat outputs;
	//Mat input1(1, 1, CV_32FC1, Scalar(0.05f));
	//Mat output1(1, 1, CV_32FC1, Scalar(0.01f));
	//Mat input2(1, 1, CV_32FC1, Scalar(0.10f));
	//Mat output2(1, 1, CV_32FC1, Scalar(0.02f));
	//inputs.push_back(input1);
	//outputs.push_back(output1);
	
	for (int i = 0; i < 100; i++) {
		Mat input(1, 1, CV_32FC1, Scalar(i * 0.01f));
		Mat output(1, 1, CV_32FC1, Scalar(i * 3 * 0.01f));
		inputs.push_back(input);
		outputs.push_back(output);
	}

	DataLayer* datalayer = new DataLayer(inputs, outputs);
	InnerProductLayer* ipLayer1 = new InnerProductLayer(3);
	InnerProductLayer* ipLayer2 = new InnerProductLayer(4);
	InnerProductLayer* ipLayer3 = new InnerProductLayer(5);
	InnerProductLayer* ipLayer4 = new InnerProductLayer(2);
	L2LossLayer* l2losslayer = new L2LossLayer(1);
	
	//L2Loss* loss = new L2Loss(1);

	net.setDataLayer(datalayer);
	net.addLayer(ipLayer1);
	net.addLayer(ipLayer2);
	net.addLayer(ipLayer3);
	net.addLayer(ipLayer4);
	net.addLayer(l2losslayer);

	net.train();

	Mat wLayer1 = ipLayer1->getW();
	Mat wLayer2 = ipLayer2->getW();
	Mat wLayer3 = ipLayer3->getW();
	Mat wLayer4 = ipLayer4->getW();
	//Mat wLossLayer = l2losslayer->getW();

	Mat test = Mat(1, 1, CV_32FC1, Scalar(0.40f));
	Mat result = net.predict(test);
	for (int i = 0; i < result.cols; i++)
		cout << result.at<float>(0, i) << endl;
	return 0;
}

