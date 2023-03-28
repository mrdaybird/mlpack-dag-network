#include <mlpack.hpp>

using namespace std;
using namespace arma;
using namespace mlpack;


// This split layer splits the input(which is only a rows atm) into multiple row matrices given by outNodes.
class Split : public Layer<mat>{
	public:
		Split(int a) : outNodes(a)
		{}

		// Takes a mat as input and reference of vector of mat as output.
		void Forward(const mat& input, std::vector<mat>& outputs){
			size_t total_rows = input.n_rows;
			if(outNodes > 1){
				size_t block_size = total_rows/outNodes;
				outputs[0] = input.rows(0, block_size - 1);
				size_t k = block_size;
				for(int i = 1; i < (outNodes-1); i++){
					outputs[i] = input.rows(k, k + block_size - 1);
					k += block_size;
				} 
				outputs[outNodes-1] = input.rows(k, input.n_rows - 1);
			}else if(outNodes == 1){
				outputs[0] = input;
			}
		}

		Layer* Clone() const{
			return new Split(outNodes);
		}
	private:
		int outNodes;
};

int main(){
	std::vector<mat> outputs(3, mat());
	std::vector<mat> actualOutputs(3, mat());

	// 'outputs' is actually mat which are created from MakeAlias, so any changes to elements of
	// 'outputs' changes the actual outputs.
	// In the real scenario outputs will contain alias matrics from 'layerOutputs' which are
	// stored in the layerOutputMatrix(see FFN for reference).
	for(int i = 0; i < 3; i++){
		actualOutputs[i].set_size(5,1);
		actualOutputs[i].fill(0);
		MakeAlias(outputs[i], actualOutputs[i].memptr(), 5, 1);
	}

	auto l2 = new Split(3);


	mat in(15, 1, fill::randn);
	std::cout << "Inputs\n" << in << std::endl;
	// Passed the input and vector of mat.
	l2->Forward(in, outputs);

	std::cout << "Outputs:" << std::endl;
	for(int i = 0; i < 3; i++){
		// actualOutputs and outputs should be same(eventhough only outputs is passed to Forward)
		std::cout << "Actual:\n" << actualOutputs[i] << std::endl;
		std::cout << "Tempororary:\n" << outputs[i] << std::endl;
	}

	return 0;
}
