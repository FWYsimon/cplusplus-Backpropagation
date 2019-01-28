#include "layer.hpp"
#include "loss.hpp"

class Net{
public:
	void setup(float learning_rate, int maxIteration, int batch_size);
	void setDataLayer(Layer* layer);
	void addLayer(Layer* layer);
	void setLoss(Loss* loss);
	void train();
	Mat predict(Mat input);

private:
	void randomData(Mat inputs, Mat labels, vector<Mat> &vecInputs, vector<Mat> &vecLabels);

private:
	vector<Layer*> allLayers;
	Loss* loss;
	float learning_rate;
	int maxIteration;
	int batch_size;
};