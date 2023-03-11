#define MLPACK_PRINT_INFO
#define MLPACK_PRINT_WARN

#include "DAGNetwork.hpp"

using namespace mlpack;
using namespace arma;
using namespace std;


class ResNet{
	public:
		ResNet(int classes, std::vector<int> _layers = {2, 2, 2, 2}) : layers(std::move(_layers)), num_classes(classes) 
		{}
		
		int Block(int inputLayer, int planes, bool downsample, std::vector<int> stride, DAGNetwork<NegativeLogLikelihood>& model){
			int x = inputLayer;
			// Each layer is represented by an unique id and Add function return the id.
			int conv3x3_1 = model.Add<Convolution>(planes,3, 3, stride[0], stride[1], 1, 1);
			int bn1 = model.Add<BatchNorm>();
			int relu1 = model.Add<ReLU>();
			int conv3x3_2= model.Add<Convolution>(planes, 3, 3, 1, 1, 1, 1);
			int bn2 = model.Add<BatchNorm>();
			
			// sequential takes in list of layer ids and create edges between successive layers
			// x -> conv3x3_1 -> bn1 -> relu1 -> conv3x3_2 -> bn2
			// sequential returns the id of the last layer such that it is easy to use in other places
			int y = model.sequential({x, conv3x3_1, bn1, relu1, conv3x3_2, bn2});
			
			if(downsample){
				int conv1x1 = model.Add<Convolution>(planes, 1, 1, stride[0], stride[1], 0, 0);
				int bn3 = model.Add<BatchNorm>();
				// x -> conv1x1 -> bn3
				x = model.sequential({x, conv1x1, bn3});
			}
			// Addition Layers takes multiple layers and adds them together.
			int add = model.Add<Addition>();
			// add_inputs takes in layer id and a vector respectively and creates directed edges from each layer ...
			// in the vector to the layer id in the first parameter. 
			// x -> add <- y
			model.add_inputs(add, {x, y});
			int relu2 = model.Add<ReLU>();
			// add -> relu2
			model.add_input(relu2, add);

			return relu2;
		}
		void createModel(){
			// The input is also given a unique id, making the input reusable.
			int x = g.InputLayer();
			int conv3x3 = g.Add<Convolution>(64, 7, 7, 2, 2, 3, 3);
			int bn1 = g.Add<BatchNorm>();
			int relu1 = g.Add<ReLU>();
			int maxpool = g.Add<MaxPooling>(3, 3, 2, 2);
			
			// x -> con3x3 -> bn1 -> relu1 -> maxpool
			x = g.sequential({x, conv3x3, bn1, relu1, maxpool});
		
			for(int i = 0; i < layers[0]; i++)
				x = Block(x, 64, false, {1, 1}, g);

			x = Block(x, 128, true, {2, 2}, g);
			for(int i = 1; i < layers[1]; i++)
				x = Block(x, 128, false, {1, 1}, g);

			x = Block(x, 256, true, {2, 2}, g);
			for(int i = 1; i < layers[2]; i++)
				x = Block(x, 256, false, {1, 1}, g);

			x = Block(x, 512, true, {2, 2}, g);
			for(int i = 1; i < layers[3]; i++)
				x = Block(x, 512, false, {1, 1}, g);

			//int avgpool = g.Add<MeanPooling>(2, 2, 2, 2);>
			int linear = g.Add<Linear>(num_classes);
			//g.sequential({x, avgpool, linear});
			int logsoftmax = g.Add<LogSoftMax>();
			// x -> linear -> logsoftmax
			x = g.sequential({x, linear, logsoftmax});
			g.OutputLayer() = x;
		}
		auto& Model(){
			if(modelCreated)
				return g;
			createModel();
			return g;
		}
	private:
		DAGNetwork<NegativeLogLikelihood> g;
		bool modelCreated = false;
		std::vector<int> layers;
		int num_classes;
};

Row<size_t> getLabels(const mat& predOut)
{
  Row<size_t> predLabels(predOut.n_cols);
  for (uword i = 0; i < predOut.n_cols; ++i)
  {
    predLabels(i) = predOut.col(i).index_max();
  }
  return predLabels;
}
	

int main(){
	mat dataX, labels;
	data::Load("data/digit-recognizer/train.csv", dataX, true, true);
	
	dataX = dataX.cols(0, 2000);
	labels = dataX.row(0);
	dataX.shed_row(0);
	dataX /= 255.0;
	mat trainX, trainY, validX, validY;
	data::Split(dataX, labels, validX, trainX, validY, trainY, 0.8);
	
	ResNet resnet18(10);
	auto& model = resnet18.Model();
	
	model.InputDimensions() = {28, 28};

	ens::RMSProp optimizer{0.01, 32, 0.99, 1e-8, 1*validX.n_cols};
	model.Train(validX, validY, optimizer, ens::ProgressBar(),  ens::Report(), ens::PrintLoss());

	mat preds = model.Predict(validX);
	auto predsLabels = getLabels(preds);
	double accuracy = accu(predsLabels == validY)/ (double)validY.n_elem * 100;
	std::cout << accuracy << std::endl;
}

