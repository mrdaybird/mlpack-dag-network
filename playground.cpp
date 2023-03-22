#include "DAGNetwork.hpp"

using namespace arma;
using namespace mlpack;

auto conv3x3(int outPlanes, int strideX = 1, int strideY = 1, int paddingX = 0, int paddingY = 0){
	return new Convolution(outPlanes, 3, 3, strideX, strideY, paddingX, paddingY);
}
auto conv1x1(int outPlanes, int strideX = 1, int strideY = 1, int paddingX = 0, int paddingY = 0){
	return new Convolution(outPlanes, 1, 1, strideX, strideY, paddingX, paddingY);
}
auto batch_norm(){
	new BatchNorm();
}

int main(){
	DAGNetwork g{};
	int x = g.InputLayer();
	
	auto conv1 = conv1x1(4);
	auto conv2 = conv3x3(4);
	g.add_input(conv1, x);
	int y = g.sequential({conv1, batch_norm(), conv2});
	g.OutputLayer() = y;

	g.InputDimensions() = {4, 4};

	mat dataX(16, 5, fill::ones);
	mat preds = g.Predict(dataX);
	std::cout << preds << std::endl;
}
