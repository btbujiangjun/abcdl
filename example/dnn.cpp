/***********************************************
 * Author: Jun Jiang - jiangjun4@sina.com
 * Create: 2017-08-19 21:06
 * Last modified : 2017-08-19 21:06
 * Filename      : dnn.cpp
 * Description   : 
 **********************************************/
#include <vector>
#include "dnn/DNN.h"
#include "utils/Log.h"
#include "utils/MnistHelper.h"

int main(int argc, char** argv){
    abcdl::utils::log::set_min_log_level(abcdl::utils::log::INFO);
    abcdl::utils::log::initialize_log(argc, argv);

    const std::string path = "./data/dnn.model";

    abcdl::dnn::DNN dnn;
    if(!dnn.load_model(path)){
        std::vector<abcdl::dnn::Layer*> layers;
        layers.push_back(new abcdl::dnn::InputLayer(784));
        layers.push_back(new abcdl::dnn::FullConnLayer(784, 30, new abcdl::framework::SigmoidActivateFunc()));
        layers.push_back(new abcdl::dnn::OutputLayer(30, 10, new abcdl::framework::SigmoidActivateFunc(), new abcdl::framework::CrossEntropyCost()));
        dnn.set_layers(layers);
    }

    abcdl::utils::MnistHelper<real> helper;
    
    abcdl::algebra::Mat train_data;
    helper.read_image("data/mnist/train-images-idx3-ubyte", &train_data, -1);

    abcdl::algebra::Mat train_label;
    helper.read_vec_label("data/mnist/train-labels-idx1-ubyte", &train_label, -1);

    abcdl::algebra::Mat test_data;
    helper.read_image("data/mnist/t10k-images-idx3-ubyte", &test_data, -1);

    abcdl::algebra::Mat test_label;
    helper.read_vec_label("data/mnist/t10k-labels-idx1-ubyte", &test_label, -1);

    dnn.train(train_data, train_label, test_data, test_label);

    dnn.write_model(path);
}
