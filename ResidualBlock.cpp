#define MLPACK_PRINT_INFO
#define MLPACK_PRINT_WARN

#include "DAGNetwork.hpp"

using namespace mlpack;
using namespace std;

int main(){
	DAGNetwork g{};
	int l1 = g.Add<Identity>();
	int l2 = g.Add<Sigmoid>();
	int l3 = g.Add<Addition>();
	g.add_inputs(l2, {l1});
	g.add_inputs(l3, {l2, l1});
	
	g.InputLayer() = l1;
	g.OutputLayer() = l3;
	arma::mat x(1, 5, arma::fill::ones);
	arma::mat y(1, 5, arma::fill::zeros);
	ens::SGD optimizer(1, 5, 1*5);
	g.Train(x, y, optimizer, ens::Report(), ens::PrintLoss());
	arma::mat d = g.getBackwardOf(1);
	std::cout << d;
	arma::mat out = g.getOutputOf(l3);
	std::cout << out;
}
