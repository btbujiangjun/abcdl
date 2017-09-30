/***********************************************
 * Author: Jun Jiang - jiangjun4@sina.com
 * Create: 2017-08-19 21:06
 * Last modified : 2017-08-30 13:43
 * Filename      : cnn.cpp
 * Description   : 
 **********************************************/
#include <vector>
#include "cnn/CNN.h"
#include "cnn/Layer.h"
#include "utils/Log.h"
#include "utils/MnistHelper.h"

int main(int argc, char** argv){
    abcdl::utils::log::set_min_log_level(abcdl::utils::log::INFO);
    abcdl::utils::log::initialize_log(argc, argv);

    const std::string path = "./data/cnn.model";

    std::vector<abcdl::cnn::Layer*> layers;
    layers.push_back(new abcdl::cnn::InputLayer(28, 28));
    layers.push_back(new abcdl::cnn::ConvolutionLayer(3, 1, 5, new abcdl::framework::SigmoidActivateFunc()));
    layers.push_back(new abcdl::cnn::SubSamplingLayer(2, new abcdl::framework::MeanPooling()));
    layers.push_back(new abcdl::cnn::ConvolutionLayer(3, 1, 5, new abcdl::framework::SigmoidActivateFunc()));
//    layers.push_back(new abcdl::cnn::SubSamplingLayer(2, new abcdl::cnn::MeanPooling()));
    layers.push_back(new abcdl::cnn::OutputLayer(10, new abcdl::framework::SigmoidActivateFunc(), new abcdl::framework::CrossEntropyCost()));

    abcdl::cnn::CNN cnn;
    cnn.set_layers(layers);

    abcdl::utils::FashionMnistReader<real> helper("data");
    const size_t train_size = 60000;
    const size_t test_size  = 10000;

    abcdl::algebra::MatSet train_data;
    helper.read_train_images(train_data, train_size, 30);

    for(size_t i = 0; i != 20 && i != train_data.rows(); i++){
        ((abcdl::algebra::IMat)train_data[i]).display("", false);
    }

    abcdl::algebra::MatSet train_label;
    helper.read_train_vec_labels(train_label, train_size);

    abcdl::algebra::MatSet test_data;
    helper.read_test_images(test_data, test_size);

    abcdl::algebra::MatSet test_label;
    helper.read_test_vec_labels(test_label, test_size);

    cnn.train(train_data, train_label, test_data, test_label);
}
