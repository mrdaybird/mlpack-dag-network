#define MLPACK_PRINT_INFO
#define STB_IMAGE_IMPLEMENTATION
#define MLPACK_PRINT_WARN

#include <mlpack.hpp>
#include <vector>
#include <map>
#include <iostream>
#include <memory>

/*
    Node of Layers that could that multiple inputs and give out multiple outputs(?)
*/

using mat = arma::mat;
using cube = arma::cube;
class Addition : public mlpack::Layer<mat>{
	private:
		size_t inSize, outSize;
	public:
	// Requires the two inputs matrices to be joined using join_cols(). Thus 'in' mat is two matrices upper and lower matrices.
	void Forward(const cube& in, mat& out){
		out = arma::sum(in, 2);
	}
	void Backward(const cube& in, const mat& gy, cube& g){
		g = arma::ones<cube>(arma::size(in));
		g.each_slice() *= gy;
	}
};

template<typename LossLayerType = mlpack::NegativeLogLikelihood>
class DAGNetwork{
    public:
		DAGNetwork(LossLayerType layer = LossLayerType()) : lossLayer(std::move(layer)){
		}
		// Block 1: Methods required by Ensmallen to act as "Differentiable separable function"
		double EvaluateWithGradient(const mat& x, const size_t i, mat& g, const size_t batch_size){
 			//Forward Pass
			layerOutputs.clear();
			layerOutputs[0] = predictors.cols(i, i + batch_size - 1);
			ForwardDAG(outputLayer);	
			
			double loss = lossLayer.Forward(layerOutputs[outputLayer], responses.cols(i, i + batch_size - 1));
			
			//Backward Pass
			error.fill(0);
			lossLayer.Backward(layerOutputs[outputLayer], responses.cols(i, i + batch_size - 1), error);
			layerBackwards.clear();
			layerBackwards[outputLayer] = error;
			gradient.fill(0);
			BackwardWithGradientDAG(inputLayer);

			g = gradient;
			return loss;
		}

		void Shuffle(){}


		size_t NumFunctions() const{
			return responses.n_cols;
		}
		// End Block 1
		int& InputLayer() {
			setInputLayer = true;
			return inputLayer;
		}
		int& OutputLayer() {
			setOutputLayer = true;
			return outputLayer;
		}			
		void ForwardDAG(int layerID){
			if(layerOutputs.find(layerID) != layerOutputs.end())
				return;
			auto& layerIn = inputs[layerID];
			for(int in : layerIn){
				ForwardDAG(in);
			}
			auto& layer = db[layerID];
			if(layerIn.size() == 1){
				mat out;
				layer->Forward(layerOutputs[layerIn[0]], out);
				layerOutputs[layerID] = std::move(out);
			}else if(layerIn.size() > 1){
				cube joined(layerIn.size(), layerOutputs[layerIn[0]].n_rows, layerOutputs[layerIn[0]].n_cols);
				for(size_t i = 1; i < layerIn.size(); i++)
					joined.slice(i) = layerOutputs[layerIn[1]];
				mat out;
				layer->Forward(joined, out);
				layerOutputs[layerID] = std::move(out);
			}
		}

		void BackwardWithGradientDAG(int layerID){
			if(layerBackwards.find(layerID) != layerBackwards.end() && layerID != outputLayer)
				return;
			const auto& layerConsumers = consumers[layerID];
			for(int c : layerConsumers){
				BackwardWithGradientDAG(c);
			}
			const auto& in = inputs[layerID];
			auto& layer = db[layerID];
			if(in.size() == 1){
				mat& input = layerOutputs[in[0]];
				mat& gy = layerBackwards[layerID];
				mat g;
				layer->Backward(input, gy, g);
				layer->Gradient(input, gy, layerGradients[layerID]);
				layerBackwards[in[0]] = g;
			}else if(in.size() > 1){/*
				cube input(in.size(), layerOutputs[in[0]].n_rows, layerOutputs[in[0]].n_cols);
				for(size_t i = 0; i < in.size(); i++)
					input.slice(i) = layerOutputs[in[i]];

				mat& gy = layerBackwards[layerID];
				cube g;
				layer->Backward(input, gy, g);
				for(size_t i = 0; i < in.size(); i++){
					if(layerBackwards.find(in[i]) == layerBackwards.end())
						layerBackwards[in[i]] = g.slice(i);
					else
						layerBackwards[in[i]] += g.slice(i);
				}*/
			}
		}

		template<typename OptimizerType, typename ...Args>
		void Train(const mat& predictors, const mat& responses, OptimizerType& optimizer, Args... callbacks){
			this->predictors = predictors;
			this->responses = responses;
			checkAndInitialize();
			optimizer.Optimize(*this, parameter, callbacks...);			
		}

		const mat& Predict(const mat& x){
			if(!setInputLayer || !setOutputLayer){
				std::cerr << "Input or Output Layer not set" << std::endl;
			}
			predictors = x;
			checkAndInitialize();
			
			layerOutputs.clear();
			layerOutputs[0] = x;

			ForwardDAG(outputLayer);
			return layerOutputs[outputLayer];
		}
		void findInputAndOutputLayers(){
			for(const auto& [id, inputVector] : inputs){
				if(inputVector.empty()){
					inputLayer = id;
					break;
				}
			}
			for(const auto& [id, consumerVector] : consumers){
				if(consumerVector.empty()){
					outputLayer = id;
					break;
				}
			}
		}

		void setInputDimensions(int layerID){
			auto layer = db[layerID];
			const auto& in = inputs[layerID];
			if(in.empty()){
				layer->InputDimensions() = {predictors.n_rows};
			}else{
				layer->InputDimensions() = db[in[0]]->OutputDimensions();
				if(in.size() >= 2)
					layer->InputDimensions().push_back(in.size());
			}
			const auto& out = consumers[layerID];
			for(int i : out){
				setInputDimensions(i);
			}
		}
		void findWeightSize(){
			weightSize = 0;
			for(const auto&[id, layer] : db){
				weightSize += layer->WeightSize();
			}
		}
		size_t WeightSize(){
			return weightSize;
		}
		void SetLayerMemory(){
			std::priority_queue<int> q;
			std::set<int> visited;
			visited.insert(inputLayer);
			q.push(inputLayer);
			size_t start = 0;
			auto ptr = parameter.memptr();
			auto gradientptr = gradient.memptr();
			while(!q.empty()){
				int top = q.top();
				q.pop();
				auto& layer = db[top];
					size_t size = layer->WeightSize();
					//std::cout << size << ' ' << start << ' ' << weightSize << std::endl;
					assert(size + start <= weightSize);
					layer->SetWeights(ptr + start);
					mlpack::MakeAlias(layerGradients[top], gradientptr + start, size, 1);	
					start += size; 
					auto& cons = consumers[top];
					for(int i : cons){
						if(visited.find(i) == visited.end()){
							visited.insert(i);
							q.push(i);
						}
					}
			}
		}
		void checkAndInitialize(){
			//findInputAndOutputLayers();
			if(checkDone == true)
				return;
			weightSize = 0;
			setInputDimensions(inputLayer);
			db[outputLayer]->ComputeOutputDimensions();
			findWeightSize();
			parameter = arma::randu(weightSize, 1);
			parameter *= 2;
			parameter -= 1;
			gradient.zeros(weightSize, 1); 
			SetLayerMemory();	
			inputs[inputLayer].push_back(0);
			checkDone = true;
		}

        template<typename LayerType, typename... Args>
        int Add(Args... args){
            int id = getid();
            auto layer = new LayerType(args...);
            db[id] = layer;
			inputs[id] = {};
			consumers[id] = {};
            return id;
        }
        void add_inputs(int node, std::vector<int> in){
            for(const auto i : in){
                inputs[node].push_back(i);
                consumers[i].push_back(node);
            }
        }
       	~DAGNetwork(){
			for(auto&[id, layer] : db){
				delete layer;
			}
		}
    private:
        int getid(){
            return ++layers;
        }    

		size_t weightSize;
        std::unordered_map<int, std::vector<int>> consumers;
        std::unordered_map<int, std::vector<int>> inputs;
        std::unordered_map<int, mlpack::Layer<mat>*> db;
		std::unordered_map<int, mat> layerOutputs;
		std::unordered_map<int, mat> layerBackwards;
		std::unordered_map<int, mat> layerGradients;
		int inputLayer, outputLayer;
		LossLayerType lossLayer;
		bool setInputLayer = false;
		bool setOutputLayer = false;
		mat parameter, gradient;
		mat error;
		mat predictors, responses;
        int layers = 0;
		bool checkDone = false;
};
