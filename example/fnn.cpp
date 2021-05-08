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
    fnn.set_alpha(0.1);
	fnn.set_batch_size(64);
    /*
    std::vector<abcdl::fnn::Layer*> layers;
    layers.push_back(new abcdl::fnn::InputLayer(784));
    layers.push_back(new abcdl::fnn::FullConnLayer(784, 128, new abcdl::framework::ReluActivateFunc()));
    layers.push_back(new abcdl::fnn::FullConnLayer(128, 128, new abcdl::framework::ReluActivateFunc()));
    layers.push_back(new abcdl::fnn::FullConnLayer(128, 64, new abcdl::framework::ReluActivateFunc()));
    layers.push_back(new abcdl::fnn::FullConnLayer(64, 32, new abcdl::framework::ReluActivateFunc()));
    layers.push_back(new abcdl::fnn::FullConnLayer(32, 16, new abcdl::framework::ReluActivateFunc()));
    layers.push_back(new abcdl::fnn::OutputLayer(16, 10, new abcdl::framework::SigmoidActivateFunc(), new abcdl::framework::CrossEntropyCost()));
    fnn.set_layers(layers);
    fnn.load_model(path);
    */
    std::vector<abcdl::fnn::Layer*> layers;
    layers.push_back(new abcdl::fnn::InputLayer(784));
    //layers.push_back(new abcdl::fnn::FullConnLayer(784, 30, new abcdl::framework::SigmoidActivateFunc()));
    layers.push_back(new abcdl::fnn::FullConnLayer(784, 128, new abcdl::framework::ReluActivateFunc()));
    layers.push_back(new abcdl::fnn::FullConnLayer(128, 30, new abcdl::framework::ReluActivateFunc()));
    layers.push_back(new abcdl::fnn::OutputLayer(30, 10, new abcdl::framework::SigmoidActivateFunc(), new abcdl::framework::CrossEntropyCost()));
    fnn.set_layers(layers);

    const int train_size = -1;
    const int test_size = -1;
    
    //abcdl::utils::FashionMnistReader<real> helper("data");
    abcdl::utils::MnistReader<real> helper("data");
    abcdl::algebra::Mat train_data;
    helper.read_train_image(&train_data, train_size);
    abcdl::algebra::Mat train_label;
    helper.read_train_vec_label(&train_label, train_size);

    abcdl::algebra::Mat test_data;
    helper.read_test_image(&test_data, test_size);
    abcdl::algebra::Mat test_label;
    helper.read_test_vec_label(&test_label, test_size);
    real loss = 0;
	int epoch = 100;
    for(int i = 0; i < epoch; i++){
		printf("Epoch:[%d/%d]\n", i, epoch);
        fnn.train(train_data, train_label);
        fnn.evaluate(test_data, test_label, &loss);
        fnn.write_model(path);
    }
}
