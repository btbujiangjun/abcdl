/***********************************************
 * Author: Jun Jiang - jiangjun4@sina.com
 * Create: 2017-08-19 21:06
 * Last modified : 2017-08-19 21:06
 * Filename      : fnn.cpp
 * Description   : 
 **********************************************/
#include <vector>
#include "fnn/FNN.h"
#include "utils/Log.h"
#include "utils/MnistHelper.h"

int main(int argc, char** argv){
    abcdl::utils::log::set_min_log_level(abcdl::utils::log::INFO);
    abcdl::utils::log::initialize_log(argc, argv);

    const std::string path = "./data/fnn.model";

    abcdl::fnn::FNN fnn;
    if(!fnn.load_model(path)){
        std::vector<abcdl::fnn::Layer*> layers;
        layers.push_back(new abcdl::fnn::InputLayer(784));
        layers.push_back(new abcdl::fnn::FullConnLayer(784, 30, new abcdl::framework::SigmoidActivateFunc()));
        layers.push_back(new abcdl::fnn::OutputLayer(30, 10, new abcdl::framework::SigmoidActivateFunc(), new abcdl::framework::CrossEntropyCost()));
        fnn.set_layers(layers);
    }

    abcdl::utils::FashionMnistReader<real> helper("data");
   
    const int train_size = -1;
    const int test_size = -1;

    abcdl::algebra::Mat train_data;
    helper.read_train_image(&train_data, train_size);
    abcdl::algebra::Mat train_label;
    helper.read_train_vec_label(&train_label, train_size);

    abcdl::algebra::Mat test_data;
    helper.read_test_image(&test_data, test_size);
    abcdl::algebra::Mat test_label;
    helper.read_test_vec_label(&test_label, test_size);

    fnn.train(train_data, train_label, test_data, test_label);

    fnn.write_model(path);
}
