/*
	Tried to replicate the first example from ANN tutorial. To see the original code: https://github.com/mlpack/mlpack/blob/master/doc/tutorials/ann.md
*/


#define MLPACK_PRINT_INFO
#define MLPACK_PRINT_WARN

#include <iostream>

#include "DAGNetwork.hpp"

using namespace mlpack;
using namespace std;


int main(){
	arma::mat trainData;
	data::Load("data/thyroid_train.csv", trainData, true);
	arma::mat testData;
	data::Load("data/thyroid_test.csv", testData, true);

	arma::mat trainLabels = trainData.row(trainData.n_rows - 1) - 1;
	arma::mat testLabels = testData.row(testData.n_rows - 1) - 1;
	trainData.shed_row(trainData.n_rows - 1);
	testData.shed_row(testData.n_rows - 1);


	/*
		The Add method returns a unique id (which is a int at the moment but I am thinking of 
						changing it to struct called LayerID, which is inspired by https://floooh.github.io/2018/06/17/handles-vs-pointers.html)
		A layer could be connected to another layer using add_inputs which takes the layer id as first parameter and second parameter
		as the std::vector<int> of input layer ids.
	*/
    DAGNetwork g{};
	int x = g.InputLayer();
    int linear1 = g.Add<Linear>(8);
	int sigmoid = g.Add<Sigmoid>();
	int linear2 = g.Add<Linear>(3);
	int logsoftmax = g.Add<LogSoftMax>();
	/* 
		The layer corresponding to the first parameter is connected with layers in the std::vector in the second paramter.
		Thus output of layers in the std::vector is passed to layer corresponding to the first parameter.

		This function could be overloaded to take only single parameter in case the layer takes a single input.
	*/
	g.add_inputs(linear1, {x});
	g.add_inputs(sigmoid, {linear1});
	g.add_inputs(linear2, {sigmoid});
	g.add_inputs(logsoftmax, {linear2});

	/*
	Architecture:
		Input->Linear(8) -> Sigmoid -> Linear(3) -> LogSoftMax->Output
		Loss: NegativeLogLikelihood
	*/

	
	//This will be removed in the future, but currently present to make the code work.
	//g.InputLayer() = linear1;
	g.OutputLayer() = logsoftmax;

	ens::Adam optimizer{};
	g.Train(trainData, trainLabels, optimizer, ens::Report());
	std::cout << arma::size(trainData) << std::endl;
	mat predictionTemp = g.Predict(testData);
	arma::mat prediction = arma::zeros<arma::mat>(1, predictionTemp.n_cols);

  for (size_t i = 0; i < predictionTemp.n_cols; ++i)
  {
    prediction(i) = arma::as_scalar(arma::find(
        arma::max(predictionTemp.col(i)) == predictionTemp.col(i), 1));
  }

  size_t correct = arma::accu(prediction == testLabels);
  double classificationError = 1 - double(correct) / testData.n_cols;

  std::cout << "Classification Error for the Test set: " << classificationError << std::endl;
}
