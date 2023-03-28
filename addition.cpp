#include <mlpack.hpp>

using namespace std;
using namespace arma;
using namespace mlpack;

class Addition : public Layer<mat>{
	public:
		Addition(int a) : inNodes(a) 
		{}
	
		void Forward(const std::vector<mat>& inputs, mat& output){
			output = inputs[0];
			for(size_t i = 1; i < inNodes; i++){
				output += inputs[i];
			}
		}
		Layer* Clone() const{
			return new Addition(inNodes);
		}		
	private:
		int inNodes;
};

int main(){
	// Think of these inputs coming from different layers.
	std::vector<mat> inputs(3, mat());

	for(int i = 0; i < 3; i++){
		inputs[i] = randn(5,5);
		std::cout << i << '\n' << inputs[i] << std::endl;
	}

	// Inputs from different layers combined into a vector for it to be passed to Forward
	std::vector<mat> tempInputs(3, mat());
	for(int i = 0; i < 3; i++)
		MakeAlias(tempInputs[i], inputs[i].memptr(), 5, 5); // Prevent copy of entire matrices.(Yay!)

	mat out(5,5);

	auto sum = new Addition(3);//Specifying the number of layers to be added could be made optional.
	sum->Forward(tempInputs, out);
	
	std::cout << "Output:\n" << out;

	return 0;
}
