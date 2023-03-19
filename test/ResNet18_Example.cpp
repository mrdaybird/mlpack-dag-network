#define MLPACK_PRINT_INFO
#define MLPACK_PRINT_WARN

#include <vector>
#include <mlpack.hpp>
#include <omp.h>

using namespace std;
using namespace arma;
using namespace mlpack;

class ResNet{
	public:
		ResNet(int num, std::vector<int> _layers = {2, 2, 2, 2}) : num_classes(num), layers(std::move(_layers)){
		}
		void Block(int planes, bool downsample, std::vector<int> stride){
			auto residualBlock = new AddMerge();
			auto path1 = new MultiLayer<mat>();
			path1->Add(new Convolution(planes, 3, 3, stride[0], stride[1], 1, 1));
			path1->Add(new BatchNorm());
			path1->Add(new ReLU());
			path1->Add(new Convolution(planes, 3, 3, 1, 1, 1, 1));
			path1->Add(new BatchNorm());
			
			residualBlock->Add(path1);
			if(downsample){
				auto path2 = new MultiLayer<mat>();
				path2->Add(new Convolution(planes, 1, 1, stride[0], stride[1], 0, 0));
				path2->Add(new BatchNorm());
				residualBlock->Add(path2);
			}else{
				residualBlock->Add(new Identity());
			}
			
			model.Add(residualBlock);
			model.Add<ReLU>();
		}
		void createModel(){
			model.Add<Convolution>(64, 7, 7, 2, 2, 3, 3);
			model.Add<BatchNorm>();
			model.Add<ReLU>();
			model.Add<MaxPooling>(3, 3, 2, 2);

			for(int i = 0; i < layers[0]; i++)
				Block(64, false, {1, 1});
			
			Block(128, true, {2, 2});
			for(int i = 1; i < layers[1]; i++)
				Block(128, false, {1, 1});

			Block(256, true, {2, 2});
			for(int i = 1; i < layers[2]; i++)
				Block(256, false, {1, 1});

			Block(512, true, {2, 2});
			for(int i = 1; i < layers[3]; i++)
				Block(512, false, {1, 1});
			
			model.Add<Linear>(num_classes);
			model.Add<LogSoftMax>();
		}
		auto& Model(){
			if(createdModel)
				return model;
			createModel();
			createdModel = true;
			return model;
		}
	private:
		FFN<NegativeLogLikelihood, RandomInitialization> model{};
		bool createdModel = false;
		int num_classes;
		std::vector<int> layers;
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
	omp_set_dynamic(0);     // Explicitly disable dynamic teams
	omp_set_num_threads(6); // Use 6 threads for all consecutive parallel regions	
	
	mat dataX, labels;
	data::Load("../data/digit-recognizer/train.csv", dataX, true, true);
	
	mat data = dataX.cols(0, 2000);
	dataX.reset();
	labels = data.row(0);
	data.shed_row(0);
	data /= 255.0;
	mat trainX, trainY, validX, validY;
	data::Split(data, labels, validX, trainX, validY, trainY, 0.8);
	
	ResNet resnet18(10);
	auto& model = resnet18.Model();
	model.InputDimensions() = {28, 28};

	ens::SGD optimizer(0.01, 32, 2*trainX.n_cols);
	model.Train(trainX, trainY, optimizer, ens::PrintLoss(), ens::Report(), ens::ProgressBar());

	mat preds;
	model.Predict(validX, preds, 32);
	auto predsLabels = getLabels(preds);
	
	double accuracy = accu(predsLabels == validY)/ (double)validY.n_elem * 100;	
	std::cout << accuracy << std::endl;
	
	return 0;
}
