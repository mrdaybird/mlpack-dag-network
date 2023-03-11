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
	public:
	// Requires the two inputs matrices to be joined using join_cols(). Thus 'in' mat is two matrices upper and lower matrices.
	void Forward(const mat& in, mat& out){
		out = in.submat(0, 0, (in.n_rows/2) - 1, in.n_cols - 1) + in.submat(in.n_rows/2, 0, in.n_rows - 1, in.n_cols - 1);
	}
	void Backward(const mat& in, const mat& gy, mat& g){
		mat g1(in.n_rows, in.n_cols, arma::fill::ones);
		g1 %= gy;
		g.submat(0, 0, g.n_rows/2 - 1, g.n_cols - 1) = g1;
		g.submat(g.n_rows/2, 0, g.n_rows - 1, g.n_cols - 1) = g1;	
	}
	Addition* Clone() const{ return new Addition(*this); }
};

template<typename LossLayerType = mlpack::NegativeLogLikelihood>
class DAGNetwork{
    public:
		DAGNetwork(LossLayerType layer = LossLayerType()) : lossLayer(std::move(layer)){
			inputs[inputLayer] = {};
			consumers[inputLayer] = {};
		}
		// Block 1: Methods required by Ensmallen to act as "Differentiable separable function"
		double EvaluateWithGradient(const mat& x, const size_t i, mat& g, const size_t batch_size){
 			//Forward Pass
			InitializeForwardPassMemory(batch_size);
			layerOutputMatrix.fill(0);	
			
			for(int i = 0; i < visitedForward.size(); i++)
				visitedForward[i] = 0;
			visitedForward[inputLayer] = 1;
			layerOutputs[inputLayer] = predictors.cols(i, i + batch_size - 1);
			ForwardDAG(outputLayer);	
			double loss = lossLayer.Forward(layerOutputs[outputLayer], responses.cols(i, i + batch_size - 1));
			if(loss == 0){
//				std::cout << layerOutputs[]
			}
			//Backward Pass
			InitializeBackwardPassMemory(batch_size);
			layerDeltaMatrix.fill(0);
			lossLayer.Backward(layerOutputs[outputLayer], responses.cols(i, i + batch_size - 1), error);
			layerBackwards[outputLayer] = error;
			
			gradient.fill(0);
			for(int i = 0; i < visitedBackward.size(); i++)
				visitedBackward[i] = 0;;
			BackwardWithGradientDAG(inputLayer);
			g = gradient;
			return loss;
		}

		void Shuffle(){}


		size_t NumFunctions() const{
			return responses.n_cols;
		}
		// End Block 1
		int InputLayer() const{
			return inputLayer;
		}
		int& InputLayer() {
			setInputLayer = true;
			return inputLayer;
		}
		int OutputLayer() const{
			if(!setOutputLayer)
				std::cerr << "Not set Output Layer" << std::endl;
			return outputLayer;
		}
		int& OutputLayer() {
			setOutputLayer = true;
			return outputLayer;
		}			
		void ForwardDAG(int layerID){
			if(visitedForward[layerID] == 1)
				return;
			auto& layerIn = inputs[layerID];
			for(int in : layerIn){
				ForwardDAG(in);
			}
			auto& layer = db[layerID];
			if(layerIn.size() == 1){
				layer->Forward(layerOutputs[layerIn[0]], layerOutputs[layerID]);
			}else if(layerIn.size() > 1){
				if(arma::size(layerOutputs[layerIn[0]]) != arma::size(layerOutputs[layerIn[1]])){
					for(int i = 0; i < db[layerIn[0]]->OutputDimensions().size(); i++)
						std::cout << db[layerIn[0]]->OutputDimensions()[i];
					std::cout << std::endl;
					for(int i = 0; i < db[layerIn[0]]->OutputDimensions().size(); i++)
						std::cout << db[layerIn[0]]->OutputDimensions()[i];
					std::cout << std::endl;
					std::runtime_error("Dimensions are not equal!");
				}
				mat joined = layerOutputs[layerIn[0]];
				//NOTE: Write it in a more optimized way
				for(size_t i = 1; i < layerIn.size(); i++)
					joined = join_cols(joined, layerOutputs[layerIn[i]]);
				layer->Forward(joined, layerOutputs[layerID]);
			}
			visitedForward[layerID] = 1;
		}

		void BackwardWithGradientDAG(int layerID){
			if(visitedBackward[layerID] == 1)
				return;
			const auto& layerConsumers = consumers[layerID];
			for(int c : layerConsumers){
				BackwardWithGradientDAG(c);
			}
			if(layerID == inputLayer){
				visitedBackward[layerID] = 1;
				return;
			}
			const auto& in = inputs[layerID];
			
			auto& layer = db[layerID];
			if(in.size() == 1){
				mat& input = layerOutputs[in[0]];
				mat& out = layerOutputs[layerID];
				mat& gy = layerBackwards[layerID];
				mat g;
				g.set_size(arma::size(layerBackwards[in[0]]));
				layer->Backward(out, gy, g);
				layer->Gradient(input, gy, layerGradients[layerID]);
				layerBackwards[in[0]] += g;
			}else if(in.size() == 2){
				mat input = join_cols(layerOutputs[in[0]], layerOutputs[in[1]]);
				mat& out = layerOutputs[layerID];
				mat& gy = layerBackwards[layerID];
				mat g(input.n_rows, input.n_cols);
				layer->Backward(out, gy, g);
				layer->Gradient(input, gy, layerGradients[layerID]);
				layerBackwards[in[0]] += g.submat(0, 0, input.n_rows/2 - 1, input.n_cols - 1);
				layerBackwards[in[1]] += g.submat(input.n_rows/2, 0, input.n_rows - 1, input.n_cols - 1);	
						
			}else if(in.size() > 2){
				//Not implemented
			}
			visitedBackward[layerID] = 1;
		}

		template<typename OptimizerType, typename ...Args>
		void Train(const mat& predictors, const mat& responses, OptimizerType& optimizer, Args... callbacks){
			this->predictors = predictors;
			this->responses = responses;
			if(inputDimensions.size() == 0){
				inputDimensions = {predictors.n_rows};
			}
			checkAndInitialize();
			setTrainingMode(true);
			optimizer.Optimize(*this, parameter, callbacks...);			
		}

		const mat& Predict(const mat& x){
			if(!setInputLayer || !setOutputLayer){
				std::cerr << "Input or Output Layer not set" << std::endl;
			}
			predictors = x;
			if(inputDimensions.size() == 0){
				inputDimensions = {x.n_rows};
			}
		
			checkAndInitialize();
			setTrainingMode(false);
			InitializeForwardPassMemory(x.n_cols);
			layerOutputMatrix.fill(0);
			for(int i = 0; i < visitedForward.size(); i++)
				visitedForward[i] = 0;
			visitedForward[inputLayer] = 1;

			layerOutputs[inputLayer] = x;	
			ForwardDAG(outputLayer);
			return layerOutputs[outputLayer];
		}
		void setTrainingMode(bool value){
			for(auto&[id, layer] : db){
				layer->Training() = value;
			}
		}
		void InitializeForwardPassMemory(size_t batchSize){
			if(batchSize * totalOutputSize > layerOutputMatrix.n_elem || batchSize * totalOutputSize < std::floor(0.1*layerOutputMatrix.n_elem)){
				layerOutputMatrix = mat(1, batchSize * totalOutputSize);
			}
			size_t start = 0;
			size_t layerOutputSize = inSize;
			mlpack::MakeAlias(layerOutputs[inputLayer], layerOutputMatrix.colptr(start), layerOutputSize, batchSize);
			start += layerOutputSize * batchSize;	
			for(auto&[id, layer] : db){
				const size_t layerOutputSize = layer->OutputSize();
				mlpack::MakeAlias(layerOutputs[id], layerOutputMatrix.colptr(start), layerOutputSize, batchSize);
				start += layerOutputSize * batchSize;
			}
		}	
		void InitializeBackwardPassMemory(size_t batchSize){
			if(batchSize * totalInputSize > layerDeltaMatrix.n_elem || batchSize * totalInputSize < std::floor(0.1*layerDeltaMatrix.n_elem)){
				layerDeltaMatrix = mat(1, batchSize * totalInputSize);
			}
			size_t start = 0;
			for(auto&[id, layer] : db){
				if(id == outputLayer) continue;
				//size_t layerInputSize = 1;
				//for(size_t i = 0; i < layer->InputDimensions().size(); i++)
				//	layerInputSize *= layer->InputDimensions()[i];
				const size_t layerInputSize = layer->OutputSize();
				mlpack::MakeAlias(layerBackwards[id], layerDeltaMatrix.colptr(start), layerInputSize, batchSize);
				start += layerInputSize * batchSize;
			}	
			size_t layerOutputSize = inSize;
			mlpack::MakeAlias(layerBackwards[inputLayer], layerDeltaMatrix.colptr(start), layerOutputSize, batchSize);
			start += layerOutputSize * batchSize;
		}
		const auto& getOutputOf(int layerID){
			return layerOutputs[layerID];	
		}
		const auto& getBackwardOf(int layerID){
			return layerBackwards[layerID];
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
		auto& InputDimensions(){
			return inputDimensions;
		}
		void setInputDimensions(int layerID){
			auto layer = db[layerID];
			const auto& in = inputs[layerID];
			if(layerID == inputLayer){
				layer->InputDimensions() = inputDimensions;
			}else{
				layer->InputDimensions() = db[in[0]]->OutputDimensions();
			//	if(in.size() >= 2)
			//		layer->InputDimensions().push_back(in.size());
			}
			const auto& out = consumers[layerID];
			for(int i : out){
				setInputDimensions(i);
			}
		}
		void ComputeOutputDimensions(int layerID, std::vector<int>& visited){
			if(visited[layerID] == 1) return;
			auto& inputLayers = inputs[layerID];
			for(int in : inputLayers){
				ComputeOutputDimensions(in, visited);
			}
			if(inputLayers.size() == 0){
				std::cerr << "Set Input Dimensions of" << layerID << std::endl;
			}else{
				std::vector<size_t> prevOutputDimensions;
				if(inputLayers[0] == inputLayer)
					prevOutputDimensions = inputDimensions;
				else
					prevOutputDimensions = db[inputLayers[0]]->OutputDimensions();
				db[layerID]->InputDimensions() = prevOutputDimensions;	
			}
			size_t layerInputSize = db[layerID]->InputDimensions()[0];
			for(size_t i = 1; i < db[layerID]->InputDimensions().size(); i++)
				layerInputSize *= db[layerID]->InputDimensions()[i];
			totalInputSize += layerInputSize;
			db[layerID]->ComputeOutputDimensions();
			size_t layerOutputSize = db[layerID]->OutputDimensions()[0];
			for(size_t i = 1; i < db[layerID]->OutputDimensions().size(); i++)
				layerOutputSize *= db[layerID]->OutputDimensions()[i];
			totalOutputSize += layerOutputSize;
			
			visited[layerID] = 1;

		}
		void ComputeOutputDimensions(){
			std::vector<int> visited(layers+1, 0);
			visited[inputLayer] = 1;
			totalOutputSize = 0;
			totalInputSize = 0;
			size_t layerOutputSize = inputDimensions[0];
			for(size_t i = 1; i < inputDimensions.size(); i++)
				layerOutputSize *= inputDimensions[i];
			inSize = layerOutputSize;
			totalOutputSize += layerOutputSize;
			totalInputSize += layerOutputSize;
			ComputeOutputDimensions(outputLayer, visited);
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
			for(int i : consumers[inputLayer]){
				q.push(i);
				visited.insert(i);
			}
			size_t start = 0;
			auto param_ptr = parameter.memptr();
			auto gradientptr = gradient.memptr();
			while(!q.empty()){
				int top = q.top();
				q.pop();
				auto& layer = db[top];
				size_t size = layer->WeightSize();
				assert(size + start <= weightSize);
				layer->SetWeights(param_ptr + start);
				mat Wtemp;
				mlpack::MakeAlias(Wtemp, param_ptr + start, size, 1);
				layer->CustomInitialize(Wtemp, size);
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
			//setInputDimensions(inputLayer);
			ComputeOutputDimensions();
			findWeightSize();
			parameter = arma::randu(weightSize, 1);
			parameter *= 2;
			parameter -= 1;
			gradient.zeros(weightSize, 1); 
			SetLayerMemory();	
			visitedForward.resize(layers + 1, 0);
			visitedBackward.resize(layers + 1, 0);
			checkDone = true;
		}

        template<typename LayerType, typename... Args>
        int Add(Args... args){
            int id = getid();
            auto layer = new LayerType(args...);
            db[id] = layer;
			inputs[id] = {};
			consumers[id] = {};
			layerOutputs[id] = mat();
			layerBackwards[id] = mat();
            return id;
        }
        void add_inputs(int node, std::vector<int> in){
            for(const auto i : in){
                inputs[node].push_back(i);
                consumers[i].push_back(node);
            }
        }
		void add_input(int source, int destination){
			inputs[source].push_back(destination);
			consumers[destination].push_back(source);
		}
		// Ordered pair of layers creates directed edge from first->second, i.e. second takes input from first
		void add_edges(std::pair<int, int> e){
			inputs[e.second].push_back(e.first);
			consumers[e.first].push_back(e.second);
		}
		template<typename T, typename... Ts>
		void add_edges(const T x, const Ts... xs){
			add_edges(x);
			add_edges(xs...);
		}
		template<typename... Ts>
		void add_edges(const Ts (&...x)[2]){
			add_edges(std::make_pair(x[0], x[1])...);
		}
		int sequential(std::vector<int> list){
			for(size_t i = 1; i < list.size(); i++)
				add_input(list[i], list[i-1]); 
			return list.back();
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

		// NOTE: These unordered_map should be replaced with vector.  
        std::unordered_map<int, std::vector<int>> consumers;
        std::unordered_map<int, std::vector<int>> inputs;
        std::unordered_map<int, mlpack::Layer<mat>*> db;
		mat layerOutputMatrix;
		size_t totalOutputSize = 0;
		mat layerDeltaMatrix;
		size_t totalInputSize = 0;
		size_t inSize = 0;
		std::unordered_map<int, mat> layerOutputs;
		std::unordered_map<int, mat> layerBackwards;
		std::unordered_map<int, mat> layerGradients;
		std::vector<int> visitedBackward;
		std::vector<int> visitedForward;
		std::vector<size_t> inputDimensions;
		int inputLayer = 0, outputLayer;
		LossLayerType lossLayer;
		bool setInputLayer = true;
		bool setOutputLayer = false;
		mat parameter, gradient;
		mat error;
		mat predictors, responses;
        int layers = 0;
		bool checkDone = false;
};
