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
#include "utils/LibsvmHelper.h"

class SessionQ{
public:

    void init(const size_t feature_dim, const size_t label_dim){
        fnn.set_alpha(0.05);
        fnn.set_batch_size(1024);
        /*
        std::vector<abcdl::fnn::Layer*> layers;
        layers.push_back(new abcdl::fnn::InputLayer(feature_dim));
        layers.push_back(new abcdl::fnn::FullConnLayer(feature_dim, 256, new abcdl::framework::ReluActivateFunc()));
        layers.push_back(new abcdl::fnn::FullConnLayer(256, 256, new abcdl::framework::ReluActivateFunc()));
        //layers.push_back(new abcdl::fnn::FullConnLayer(256, 256, new abcdl::framework::ReluActivateFunc()));
        //layers.push_back(new abcdl::fnn::FullConnLayer(256, 64, new abcdl::framework::ReluActivateFunc()));
        layers.push_back(new abcdl::fnn::FullConnLayer(256, 32, new abcdl::framework::ReluActivateFunc()));
        layers.push_back(new abcdl::fnn::OutputLayer(32, label_dim, new abcdl::framework::SigmoidActivateFunc(), new abcdl::framework::CrossEntropyCost()));
        fnn.set_layers(layers);
        */
        std::vector<abcdl::fnn::Layer*> layers;
        layers.push_back(new abcdl::fnn::InputLayer(feature_dim));
        layers.push_back(new abcdl::fnn::FullConnLayer(feature_dim, 256, new abcdl::framework::ReluActivateFunc()));
        layers.push_back(new abcdl::fnn::FullConnLayer(256, 64, new abcdl::framework::ReluActivateFunc()));
        layers.push_back(new abcdl::fnn::OutputLayer(64, label_dim, new abcdl::framework::SigmoidActivateFunc(), new abcdl::framework::CrossEntropyCost()));
        fnn.set_layers(layers);

        const std::string path = "./data/sessionq.model";
        fnn.load_model(path);
    }

    void pass(abcdl::algebra::Mat train_data,
              abcdl::algebra::Mat train_label,
              abcdl::algebra::Mat test_data,
              abcdl::algebra::Mat test_label){
        real loss = 0;
        fnn.train(train_data, train_label);
        fnn.evaluate(test_data, test_label, &loss);
        fnn.write_model();
    }

private:
    abcdl::fnn::FNN fnn;
};//class SessionQ

int main(int argc, char** argv){
    abcdl::utils::log::set_min_log_level(abcdl::utils::log::INFO);
    abcdl::utils::log::initialize_log(argc, argv);

    abcdl::utils::LibsvmHelper<real> helper;
    abcdl::algebra::Mat train_data;
    abcdl::algebra::Mat train_label;
    abcdl::algebra::Mat test_data;
    abcdl::algebra::Mat test_label;
    size_t feature_dim = 251;
    size_t label_dim = 2;
    
    std::vector<std::string> paths;
    paths.push_back("./data/sessionq/sessionq.train.libsvmaa");
    paths.push_back("./data/sessionq/sessionq.train.libsvmab");
    paths.push_back("./data/sessionq/sessionq.train.libsvmac");
    paths.push_back("./data/sessionq/sessionq.train.libsvmad");
    paths.push_back("./data/sessionq/sessionq.train.libsvmae");
    paths.push_back("./data/sessionq/sessionq.train.libsvmaf");
    paths.push_back("./data/sessionq/sessionq.train.libsvmag");
    paths.push_back("./data/sessionq/sessionq.train.libsvmah");
    paths.push_back("./data/sessionq/sessionq.train.libsvmai");
    
    SessionQ sessionq;
    sessionq.init(feature_dim, label_dim);
    
    helper.read_data(feature_dim, label_dim, "./data/sessionq/sessionq.train.libsvmaj", &test_data, &test_label);
    for(auto& path : paths){
        helper.read_data(feature_dim, label_dim, path, &train_data, &train_label);
        sessionq.pass(train_data, train_label, test_data, test_label);
    }
}
