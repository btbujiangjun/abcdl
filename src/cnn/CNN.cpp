/*********************************************
* Author: Jun Jiang - jiangjun4@sina.com
* Created: 2017-05-31 17:22
* Last modified: 2017-08-29 18:00
* Filename: CNN.cpp
* Description: Convolution Neural Network 
**********************************************/

#include "cnn/CNN.h"

namespace abcdl{
namespace cnn{

void CNN::set_layers(std::vector<abcdl::cnn::Layer*> layers){
    CHECK(layers.size() > 0);
    CHECK(layers[0]->get_layer_type() == abcdl::algebra::INPUT);
    CHECK(layers[layers.size() - 1]->get_layer_type() == abcdl::algebra::OUTPUT);

    abcdl::cnn::Layer* pre_layer = nullptr;
    for(auto& layer : layers){
        layer->initialize(pre_layer);
        pre_layer = layer;
    }
}

void CNN::train(abcdl::algebra::BaseMatrixT<real>* train_data,
                abcdl::algebra::BaseMatrixT<real>* train_label,
                uint epoch,
                abcdl::algebra::BaseMatrixT<real>* test_data,
                abcdl::algebra::BaseMatrixT<real>* test_label){
    uint num_train_data = train_data->get_rows();
    uint num_test_data = 0;
    if(test_data != nullptr){
        num_test_data = test_data->get_rows();
    }

    //check cnn structure and data dim
    if(!check(train_data->get_cols())){
        return;
    }

    auto mini_batch_data = new abcdl::algebra::DenseMatrixT<real>();
    auto mini_batch_label = new abcdl::algebra::DenseMatrixT<real>();
    auto now = []{return std::chrono::system_clock::now();};

    for(uint i = 0; i != epoch; i++){

        auto start_time = now();
    	bool debug = (num_train_data < 10);
        for(uint j = 0; j != num_train_data; j++){
            train_data->get_row_data(j, mini_batch_data);
            train_label->get_row_data(j, mini_batch_label);

            feed_forward(mini_batch_data, debug);
            back_propagation(mini_batch_label, debug);

            if(j % 100 == 0){
                printf("Epoch[%d][%d/%d]training...\r", i, j, num_train_data);
            }
        }//end per epoch

        auto training_time = now();
        printf("Epoch %d training run time: %ld ms\n", i, std::chrono::duration_cast<std::chrono::milliseconds>(training_time - start_time).count());

        int cnt = 0;
    	debug = (num_test_data < 10);
        for(uint k = 0; k != num_test_data; k++){
            test_data->get_row_data(k, mini_batch_data);
            test_label->get_row_data(k, mini_batch_label);
            if(evaluate(mini_batch_data, mini_batch_label, debug)){
                cnt++;
            }
        }

	    if(num_test_data > 0){
	        printf("Epoch %d %d/%d\n", i, cnt, num_test_data);
    	    printf("Epoch %d predict run time: %ld ms\n", i, std::chrono::duration_cast<std::chrono::milliseconds>(now() - training_time).count());
	    }
    }//end all epoch

    delete mini_batch_data;
    delete mini_batch_label;
}

void CNN::feed_forward(abcdl::algebra::BaseMatrixT<real>* mat, bool debug){
    uint layer_size = _layers.size();
    for(uint k = 0; k < layer_size; k++){
        auto layer = _layers[k];
        Layer* pre_layer = nullptr;
        if(k == 0){
            mat->reshape(layer->get_rows(), layer->get_cols());
            ((DataLayer*)layer)->set_x(mat);
        }else{
            pre_layer = _layers[k - 1];
        }
        layer->feed_forward(pre_layer, debug);
    }//end feed_forward
}

void CNN::back_propagation(abcdl::algebra::BaseMatrixT<real>* mat, bool debug){
    int layer_size = _layers.size();
    for(int k = layer_size - 1; k >= 0; k--){
        auto layer = _layers[k];
        Layer* pre_layer = nullptr;
        Layer* back_layer = nullptr;
        if(k > 0){
            pre_layer = _layers[k - 1];
        }
        if(k < layer_size -1){
            back_layer = _layers[k + 1];
        }
        if(typeid(*layer) == typeid(FullConnectionLayer)){
            mat->reshape(mat->get_cols(),mat->get_rows());
            ((FullConnectionLayer*)_layers[k])->set_y(mat);
        }
        layer->back_propagation(pre_layer, back_layer, debug);
    }//end back_propagation
}

bool CNN::evaluate(abcdl::algebra::BaseMatrixT<real>* data, abcdl::algebra::BaseMatrixT<real>* label, bool debug){
    feed_forward(data, debug);
    auto layer = _layers[_layers.size() - 1];
    auto predict_mat = layer->get_activation(0);
    real max_value = 0;
    int max_idx = 0;
    uint rows = predict_mat->get_rows();

    for(uint i = 0; i != rows; i++){
        real value = predict_mat->get_data(i);
        if(i == 0 || value > max_value){
            max_value = value;
            max_idx = i;
        }
    }

    if(debug){
        predict_mat->transpose();
        predict_mat->display("|");
        predict_mat->transpose();
        if(max_idx != label->get_data(0)){
            printf("[%d][%d]\n", max_idx, static_cast<int>(label->get_data(0)));
        }
    }

    return max_idx == label->get_data(0);
}


bool CNN::check(uint size){
    if(_layers.size() <= 2){
        printf("convolution neural network layer must bemore than 2.\n");
        return false;
    }
    if(typeid(*(_layers[_layers.size() -1])) != typeid(FullConnectionLayer)){
        printf("Convolution neural network last layer must be FullConnectionLayer\n");
        return false;
    }
    if(size != _layers[0]->get_rows() * _layers[0]->get_cols()){
        printf("Data dim size error.");
        return false;
    }
    return true;
}

}//namespace cnn
}//namespace abcdl
