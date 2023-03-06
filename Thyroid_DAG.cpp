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

    DAGNetwork g{};
    int l1 = g.Add<Linear>(8);
	int l2 = g.Add<Sigmoid>();
	int l3 = g.Add<Linear>(3);
	int l4 = g.Add<LogSoftMax>();
	g.add_inputs(l2, {l1});
	g.add_inputs(l3, {l2});
	g.add_inputs(l4, {l3});
	g.InputLayer() = l1;
	g.OutputLayer() = l4;

	ens::Adam optimizer{};
	g.Train(trainData, trainLabels, optimizer, ens::Report());
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
