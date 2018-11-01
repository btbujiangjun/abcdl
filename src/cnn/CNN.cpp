/*********************************************
* Author: Jun Jiang - jiangjun4@sina.com
* Created: 2017-05-31 17:22
* Last modified: 2017-09-01 10:45
* Filename: CNN.cpp
* Description: Convolution Neural Network 
**********************************************/

#include "cnn/CNN.h"
#include "framework/Layer.h"
#include "utils/Log.h"
#include "utils/Shuffler.h"
#include <chrono>

namespace abcdl{
namespace cnn{

void CNN::set_layers(std::vector<abcdl::cnn::Layer*> layers){
    CHECK(layers.size() > 0 && layers[0]->get_layer_type() == abcdl::framework::INPUT);
    CHECK(layers[layers.size() - 1]->get_layer_type() == abcdl::framework::OUTPUT);

    abcdl::cnn::Layer* pre_layer = nullptr;
    for(auto& layer : layers){
        layer->initialize(pre_layer);
        _layers.push_back(layer);
        pre_layer = layer;
    }
}

void CNN::train(const abcdl::algebra::MatSet& train_data,
                const abcdl::algebra::MatSet& train_label,
                const abcdl::algebra::MatSet& test_data,
                const abcdl::algebra::MatSet& test_label){
	LOG(INFO) << "cnn train starting...";

	CHECK(train_data.size() == train_label.size() &&
       test_data.size() == test_label.size() &&
       train_data.rows() == test_data.rows() &&
       train_data.cols() == test_data.cols());
    if(!check(train_data.rows(), train_data.cols())){
        return;
    }

    size_t num_train_data = train_data.size();
    size_t num_test_data = test_data.size();
    abcdl::utils::Shuffler shuffler(num_train_data);
    auto now = []{return std::chrono::system_clock::now();};

    for(size_t i = 0; i != _epoch; i++){
        auto start_time = now();
        shuffler.shuffle();

        for(size_t j = 0; j != num_train_data; j++){
            forward(train_data[shuffler.get(j)]);
            backward(train_label[shuffler.get(j)]);

            if(j % _batch_size  == _batch_size - 1 || j == num_train_data - 1){
                update_gradient(j % _batch_size + 1, _alpha);
            }

            if(j % 100 == 0){
                printf("Epoch[%ld][%ld/%ld]training...\r", i, j, num_train_data);
            }
        }//end per epoch

        auto training_time = now();
        printf("Epoch[%ld] train run time: [%ld] ms\n", i, std::chrono::duration_cast<std::chrono::milliseconds>(training_time - start_time).count());

	    if(num_test_data > 0){
            printf("Epoch[%ld] [%ld/%ld] predict run time:[%ld] ms\n", i, evaluate(test_data, test_label), num_test_data, std::chrono::duration_cast<std::chrono::milliseconds>(now() - training_time).count());
	    }
    }
}

void CNN::predict(abcdl::algebra::Mat& result, const abcdl::algebra::Mat& predict_data){
    forward(predict_data);
    result = _layers[_layers.size() - 1]->get_activation(0);
}

void CNN::forward(const abcdl::algebra::Mat& mat){
    size_t layer_size = _layers.size();
    abcdl::cnn::Layer* pre_layer = nullptr;

    for(size_t k = 0; k != layer_size; k++){
        auto layer = _layers[k];
        if(layer->get_layer_type() == abcdl::framework::INPUT){
            ((InputLayer*)layer)->set_x(mat);
        }

        layer->forward(pre_layer);
        pre_layer = layer;
    }
}

void CNN::backward(const abcdl::algebra::Mat& mat){
    int layer_size = _layers.size();
    abcdl::cnn::Layer* back_layer = nullptr;
    abcdl::cnn::Layer* pre_layer = nullptr;

    for(int k = layer_size - 1; k >= 0; k--){
        auto layer = _layers[k];
        if(layer->get_layer_type() == abcdl::framework::OUTPUT){
            ((OutputLayer*)layer)->set_y(mat.clone().reshape(mat.cols(), mat.rows()));
        }

        pre_layer = (k > 0) ? _layers[k - 1] : nullptr;
        layer->backward(pre_layer, back_layer);
        back_layer = layer;
    }
}

void CNN::update_gradient(const size_t batch_size, const real alpha){
    for(auto& layer : _layers){
        layer->update_gradient(batch_size, alpha);
    }
}

size_t CNN::evaluate(const abcdl::algebra::MatSet& data_mat, const abcdl::algebra::MatSet& label_mat){
    size_t cnt = 0;
    size_t size = data_mat.size();
    for(size_t i = 0; i != size; i++){
        forward(data_mat[i]);
        if(label_mat[i].argmax() == _layers[_layers.size() - 1]->get_activation(0).argmax()){
            cnt++;
        }
    }
    return cnt;
}


bool CNN::check(const size_t rows, const size_t cols) const{
    if(_layers.size() <= 2){
        printf("layer size[%ld]\n", _layers.size());
        LOG(FATAL) << "convolution neural network layer must be more than 2";
        return false;
    }
    if(_layers[_layers.size() - 1]->get_layer_type() != abcdl::framework::OUTPUT){
        LOG(FATAL) << "Convolution neural network last layer must be OutputLayer";
        return false;
    }
    if(rows != _layers[0]->get_rows() || cols != _layers[0]->get_cols()){
        LOG(FATAL) << "Data dim size error";
        return false;
    }
    return true;
}

}//namespace cnn
}//namespace abcdl
