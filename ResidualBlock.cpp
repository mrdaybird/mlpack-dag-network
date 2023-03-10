//NOTE: This does not work at the momemt. :(, hopefully it works soon.
// This is a prototype of the api of DAGNetwork. 

#define MLPACK_PRINT_INFO
#define MLPACK_PRINT_WARN

#include "DAGNetwork.hpp"

using namespace mlpack;
using namespace std;

int ResidualBlock(DAGNetwork& model, int inputLayer, std::vector<size_t> channels){
	if(channels.size() != 3){
		std::cout << "Please give 3 channel sizes" << std::endl;
		return;
	}
	int x = inputLayer;
	int conv3x3 = model.Add<Convolution>(channels[0], 3, 3, 2, 2, 1, 1);
	int bn1 = model.Add<BatchNorm>(channels[0]);
	int relu = model.Add<ReLU>();
	int conv1x1_1 = model.Add<Convolution>(channels[1], 1, 1, 2, 2, 0, 0);
	int bn2 = model.Add<BatchNorm>(channels[1]);

	// Created directed edges: 
	// x-> con3x3-> bn1-> relu-> conv1x1_1-> bn2
	model.sequential({x, conv3x3, bn1, relu, conv1x1_1, bn2});
	// This could allow creating the layers inplace(i.e. inside the sequential method
	
	int conv1x1_2 = model.Add<Convolution>(channels[1], 1, 1, 2, 2, 0, 0);
	int bn3 = model.Add<BatchNorm>(channels[1]);
	// x-> conv1x1_2 -> bn3
	model.sequential({x, conv1x1_2, bn3});
	
	// Addition is a layer not yet define, but it add output of different layers, and performs backward pass for each of the input layers. At the moment this is the only solution I could think of, we would more layers like this to support few different functions.
	int addlayers = model.Add<Addition>();
	// Connects the 'bn2' and 'bn2' to the 'addLayer' layer.
	model.add_inputs(addLayers, {bn2, bn3});
	
	int relu_1 = model.Add<ReLU>();
	// addLayers -> relu_1
	model.add_input(relu_1, addlayers);

	// Return the last layer that can be connected to other layers or create another residual blocks.
	return relu_1;
}


int main(){
	DAGNetwork g{};
	int x = g.InputLayer();
	//Channels Block1:(1->4->8) Block2:(8->16->32)
	x = ResidualBlock(g, x, {4, 8});
	x = ResidualBlock(g, x, {16, 32});
	
	int linear1 = g.Add<Linear>(10);
	g.add_input(linear_1, x);
}
