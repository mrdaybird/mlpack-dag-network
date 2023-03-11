#include "DAGNetwork.hpp"

using namespace mlpack;
using namespace arma;

int main(){
	DAGNetwork g{};
	int x = g.InputLayer();
	int linear1 = g.Add<Linear>(2);
	int sigmoid = g.Add<Sigmoid>();
	g.add_edges({x, linear1}, {linear1, sigmoid});
	g.OutputLayer() = sigmoid;
	mat dataX(5,5, fill::randn);
	mat dataY(1,5, fill::randn);
	
	ens::SGD optimizer;
	g.Train(dataX, dataY, optimizer, ens::PrintLoss(), ens::Report());

	mat data(5,5, fill::ones);
	mat preds = g.Predict(data);
	std::cout << preds << std::endl;
}
