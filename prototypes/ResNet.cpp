#include <vector>
#include <mlpack.hpp>

using namespace mlpack;

typedef struct LayerNode {} LayerNode;
class Addition;

class ResNet{
    public: 

    ResNet(int num_classes, std::vector<int> layers = {2,2,2,2}) : 
         num_classes(num_classes), layers(std::move(layers))
    {}

    //utitilities
    auto conv3x3(int outPlanes, int strideX = 1, int strideY = 1, int paddingX = 0, int paddingY = 0){
	return new Convolution(outPlanes, 3, 3, strideX, strideY, paddingX, paddingY);
    }
    auto conv1x1(int outPlanes, int strideX = 1, int strideY = 1, int paddingX = 0, int paddingY = 0){
        return new Convolution(outPlanes, 1, 1, strideX, strideY, paddingX, paddingY);
    }
    auto batch_norm(){
        new BatchNorm();
    }
    auto relu(){
        new ReLU();
    }
    // end utilities

    LayerNode Block(LayerNode inputNode, int out_planes, bool downsample = false, std::vector<int> stride = {2, 2}, NN& model)
    {
        auto x = inputNode;

        LayerNode y = model.createSequential({
            conv3x3(out_planes, stride[0], stride[1], 1, 1),
            batch_norm(),
            relu(),
            conv3x3(out_planes, 1, 1, 1, 1),
            batch_norm()
        }, x);

        if(downsample){
            x = model.createSequential({
                conv1x1(out_planes, stride[0], stride[1]),
                batch_norm()
            }, x);
        }

        LayerNode add = model.Add(new Addition(), {y, x});
        LayerNode relu1 = model.Add(relu(), add);

        return relu1;
    }

    void createModel(){
        auto x = model.Input();

        x = model.createSequential({
            new Convolution(64, 7, 7, 2, 2, 3, 3),
            batch_norm(),
            relu(),
            new MaxPooling(3, 3, 2, 2)
        }, x);

        x = Block(x, 64, false, {1, 1}, model);
        for(int i = 1; i < layers[0]; i++)
            x = Block(x, 64, false, {1, 1}, model);
        
        x = Block(x, 128, true, {2, 2}, model);
        for(int i = 1; i < layers[1]; i++)
            x = Block(x, 128, false, {1, 1}, model);

        x = Block(x, 256, true, {2, 2}, model);
        for(int i = 1; i < layers[2]; i++)
            x = Block(x, 256, false, {1, 1}, model);

        x = Block(x, 512, true, {2, 2}, model);
        for(int i = 1; i < layers[3]; i++)
            x = Block(x, 512, false, {1, 1}, g);
        
        x = model.createSequential({
            new Linear(num_classes),
            new LogSoftMax()
        }, x);
        model.OutputLayer() = x;
    }

    auto& Model(){
        if(!modelCreted){
            createModel();
            modelCreated = true;
        }
        return model;
    }
    private:

    NN model;
    bool modelCreated = false;
    int num_classes;
    std::vector<int> layers;
};