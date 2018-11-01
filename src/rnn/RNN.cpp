/***********************************************
 * Author: Jun Jiang - jiangjun4@sina.com
 * Create: 2017-09-05 16:06
 * Last modified : 2017-09-05 16:06
 * Filename      : RNN.cpp
 * Description   : 
 **********************************************/

#include "rnn/RNN.h"
#include "utils/Log.h"
#include "utils/Shuffler.h"
#include <functional>

namespace abcdl{
namespace rnn{

void RNN::train(const abcdl::algebra::MatSet& train_seq_data,
                const abcdl::algebra::MatSet& train_seq_label){
	if(!check_data(train_seq_data, train_seq_label)){
        printf("RNN::sgd Data dim Error.\n");
		return ;
	}
	
	size_t num_train_data = train_seq_data.size();
    abcdl::utils::Shuffler shuffler(num_train_data);
	auto now = []{return std::chrono::system_clock::now();};

	abcdl::algebra::Mat state;
	abcdl::algebra::Mat activation;
	
	abcdl::algebra::Mat batch_derivate_weight(_U.rows(), _U.cols());
	abcdl::algebra::Mat batch_derivate_pre_weight(_W.rows(), _W.cols());
	abcdl::algebra::Mat batch_derivate_act_weight(_V.rows(), _V.cols());

	for(size_t i = 0; i != _epoch; i++){
        shuffler.shuffle();
		auto start_time = now();

		for(size_t j = 0; j != num_train_data; j++){
			_layer->farward(train_seq_data[shuffler.get(j)], _U, _W, _V, state, activation);
			_layer->backward(train_seq_data[shuffler.get(j)], train_seq_label[shuffler.get(j)], _U, _W, _V, state, activation, batch_derivate_weight, batch_derivate_pre_weight, batch_derivate_act_weight);

            if( j % _mini_batch_size == (_mini_batch_size - 1) || j == (num_train_data - 1)){
				size_t n = j % _mini_batch_size + 1;
    			_U -= batch_derivate_weight * (_alpha / n);
			    _W -= batch_derivate_pre_weight * (_alpha / n);
			    _V -= batch_derivate_act_weight * (_alpha / n);

			}

            if( j % 5 == 0){
                printf("Epoch[%ld][%ld/%ld] training...\r", i, j, num_train_data);
            }
		}

        if(_path != ""){
            write_model(_path);
        }

        printf("Epoch[%ld] training run time: %lld ms, loss[%f] base loss[%f]\n", i, (long long int)std::chrono::duration_cast<std::chrono::milliseconds>(now() - start_time).count(), loss(train_seq_data, train_seq_label), std::log(_feature_dim));
	}

	printf("training finished.\n");
}

real RNN::total_loss(const abcdl::algebra::MatSet& train_seq_data,
                     const abcdl::algebra::MatSet& train_seq_label){

	real loss_value = 0;
    abcdl::algebra::Mat state;
    abcdl::algebra::Mat activation;
	
    size_t num_train_data = train_seq_data.size();
	for(size_t j = 0; j != num_train_data; j++){
		_layer->farward(train_seq_data[j], _U, _W, _V, state, activation);

		auto mat_label = train_seq_label[j].argmax(abcdl::algebra::Axis_type::ROW);
        size_t rows = mat_label.rows();
        for(size_t row = 0; row != rows; row++){
            loss_value -= std::log(activation.get_data(row, mat_label.get_data(row, 0)));
        }
	}

	return loss_value;
}

real RNN::loss(const abcdl::algebra::MatSet& train_seq_data,
               const abcdl::algebra::MatSet& train_seq_label){

    //L(y, o) = - (1/N)(Sum y_n*log(o_n))

    real loss_value = total_loss(train_seq_data, train_seq_label);
    size_t N = 0;
    
    size_t size = train_seq_label.size();
    for(size_t i = 0; i != size; i++){
        N += train_seq_label[i].rows();
    }

    return loss_value / N;
}

bool RNN::check_data(const abcdl::algebra::MatSet& train_seq_data,
              		 const abcdl::algebra::MatSet& train_seq_label){
    return train_seq_data.size() == train_seq_label.size() &&
           train_seq_data.rows() == train_seq_label.rows() &&
           train_seq_data.cols() == train_seq_label.cols();
}			  

bool RNN::load_model(const std::string& path){
    std::vector<abcdl::algebra::Mat*> models;
    if(!loader.read<real>(path, &models,"RNNMODEL") || models.size() != 3){
        for(auto&& model : models){
            delete model;
        }
        return false;
    }
    
    _U = *(models[0]);
    _W = *(models[1]);
    _U = *(models[2]);

    _feature_dim    = _U.cols();
    _hidden_dim     = _U.rows();
    _path           = path;

    if(_layer != nullptr){
        delete _layer;
    }
    _layer = new abcdl::rnn::Layer(_hidden_dim, _bptt_truncate, new abcdl::framework::CrossEntropyCost(), new abcdl::framework::TanhActivateFunc());

    for(auto&& mat : models){
        delete mat;
    }
    models.clear();

    return true;
}

bool RNN::write_model(const std::string& path){
    std::vector<abcdl::algebra::Mat*> models;
    models.push_back(&_U);
    models.push_back(&_W);
    models.push_back(&_V);
    return loader.write<real>(models, path, "RNNMODEL", false);
}

}//namespace rnn
}//namespace abcdl
