#define MLPACK_PRINT_INFO
#define MLPACK_PRINT_WARN

#include "DAGNetwork.hpp"

using namespace mlpack;
using namespace arma;
using namespace std;

int main(){
	mat dataX, labels;
	data::Load("data/digit-recognizer/train.csv", dataX, true, true);
	
	dataX = dataX.cols(0, 7999);
	labels = dataX.row(0);
	dataX.shed_row(0);
	dataX /= 255.0;
	mat trainX, trainY, validX, validY;
	data::Split(dataX, labels, validX, trainX, validY, trainY, 0.8);
	
	std::cout << arma::size(trainX) << " " << arma::size(validX) << std::endl;

	DAGNetwork g{};
	int x = g.InputLayer();
	int conv7x7half = g.Add<Convolution>(16, 7, 7, 2, 2, 0, 0);
	int relu = g.Add<ReLU>();
	g.add_edges({x, conv7x7half}, {conv7x7half, relu});
	std::vector<int> layers;
	layers.insert(layers.end(), {x, conv7x7half, relu});
	for(int i = 0; i < 1; i++){
		int conv3x3_1 = g.Add<Convolution>(16, 3, 3, 1, 1, 1 ,1);
		int relu_1 = g.Add<ReLU>();
		int conv3x3_2 = g.Add<Convolution>(16, 3, 3, 1, 1, 1, 1);
		int relu_2 = g.Add<ReLU>();
		int addition = g.Add<Addition>();
		g.add_input(conv3x3_1, layers.back());
		g.add_edges({conv3x3_1, relu_1}, {relu_1, conv3x3_2}, {conv3x3_2, relu_2});
		g.add_inputs(addition, {relu_2, layers.back()});
		layers.insert(layers.end(), {conv3x3_1, relu_1, conv3x3_2, relu_2, addition});
	}
	int linear1 = g.Add<Linear>(10);
	int logsoftmax = g.Add<LogSoftMax>();
	g.add_edges({layers.back(), linear1}, {linear1, logsoftmax});
	g.InputDimensions() = {28, 28};
	g.OutputLayer() = logsoftmax;
	
	ens::SGD optimizer;
	g.Train(trainX,trainY, optimizer, ens::Report(), ens::PrintLoss());
}

