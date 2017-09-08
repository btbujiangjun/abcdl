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

void RNN::train(std::vector<abcdl::algebra::Mat*>* train_seq_data,
                std::vector<abcdl::algebra::Mat*>* train_seq_label){
	if(!check_data(train_seq_data, train_seq_label)){
        printf("RNN::sgd Data dim Error.\n");
		return ;
	}
	
	size_t num_train_data = train_seq_data->size();
    abcdl::utils::Shuffle shuffler(num_train_data);
	auto now = []{return std::chrono::system_clock::now();};

    std::vector<abcdl::algebra::Mat*> mini_batch_data;
    std::vector<abcdl::algebra::Mat*> mini_batch_label;

	for(size_t i = 0; i != _epoch; i++){
        shuffler.shuffle();
		auto start_time = now();

		for(size_t j = 0; j != num_train_data; j++){
            mini_batch_data.push_back(train_seq_data->at(shuffler.get_row(j)));
            mini_batch_label.push_back(train_seq_label->at(shuffler.get_row(j)));

            if( j % mini_batch_size == (mini_batch_size - 1) || j == (num_train_data - 1)){
                mini_batch_update(mini_batch_data, mini_batch_label);

                mini_batch_data.clear();
                mini_batch_label.clear();
            }

            if( j % 5 == 0){
                printf("Epoch[%d][%d/%d] training...\r", i, j, num_train_data);
            }
		}

        if(_path != ""){
            write_model(_path);
        }

        printf("Epoch[%d] training run time: %lld ms, loss[%f] base loss[%f]\n", i, (long long int)std::chrono::duration_cast<std::chrono::milliseconds>(now() - start_time).count(), loss(train_seq_data, train_seq_label), std::log(_feature_dim));
	}

	printf("training finished.\n");
}

void RNN::mini_batch_update(std::vector<abcdl::algebra::Mat*> train_seq_data,
                            std::vector<abcdl::algebra::Mat*> train_seq_label){

    const size_t num_train_data   = train_seq_data.size();
    
    std::vector<abcdl::algebra::Mat*> derivate_weight;
    std::vector<abcdl::algebra::Mat*> derivate_pre_weight;
    std::vector<abcdl::algebra::Mat*> derivate_act_weight;
    for(size_t i = 0 ; i != num_train_data; i++){
        derivate_weight.push_back(new abcdl::algebra::DenseMatrixT<real>(_U->rows(), _U->cols()));
        derivate_pre_weight.push_back(new abcdl::algebra::DenseMatrixT<real>(_W->rows(), _W->cols()));
        derivate_act_weight.push_back(new abcdl::algebra::DenseMatrixT<real>(_V->rows(), _V->cols()));
    }

    const size_t num_thread = std::min(num_train_data, _num_hardware_concurrency);

    std::vector<std::thread> threads;
    for(size_t i = 0; i != num_train_data; i++){
        threads.push_back(
            std::thread(
                std::bind(&Layer::backward, *_layer, *train_seq_data[i], *train_seq_label[i], _U, _W, _V, *derivate_weight[i], *derivate_pre_weight[i], *derivate_act_weight[i])
            )
        );
        if(i == (num_train_data - 1) || i % num_thread == (num_thread - 1) ){
            for(auto&& thread : threads){
                thread.join();
            }
            threads.clear();
        }
    }

    for(size_t i = 1; i != num_train_data; i++){
        derivate_weight[0]->add(derivate_weight[i]);
        derivate_pre_weight[0]->add(derivate_pre_weight[i]);
        derivate_act_weight[0]->add(derivate_act_weight[i]);
    }

    derivate_weight[0]->multiply(_alpha / num_train_data);
    derivate_pre_weight[0]->multiply(_alpha / num_train_data);
    derivate_act_weight[0]->multiply(_alpha / num_train_data);

    _U -= derivate_weight[0] * (_alpha / num_train_data);
    _W -= derivate_pre_weight[0] * (_alpha / num_train_data);
    _V -= derivate_act_weight[0] * (_alpha / num_train_data);

    for(auto d : derivate_weight){
        delete d;
    }
    derivate_weight.clear();
    for(auto d : derivate_pre_weight){
        delete d;
    }
    derivate_pre_weight.clear();
    for(auto d : derivate_act_weight){
        delete d;
    }
    derivate_act_weight.clear();
}

real RNN::total_loss(std::vector<abcdl::algebra::Mat*>* train_seq_data,
                     std::vector<abcdl::algebra::Mat*>* train_seq_label){

	real loss_value = 0;
    abcdl::algebra::Mat state;
    abcdl::algebra::Mat activation;
	
    size_t num_train_data = train_seq_data->size();
	for(size_t j = 0; j != num_train_data; j++){
		_layer->farward(*train_seq_data->at(j), _U, _W, _V, state, activation);

		auto mat_label = train_seq_label->at(j)->argmax(0);
        size_t rows = mat_label->rows();
        for(size_t row = 0; row != rows; row++){
            loss_value -= std::log(activation->get_data(row, mat_label->get_data(row, 0)));
        }
        delete mat_label;
	}

	return loss_value;
}

real RNN::loss(std::vector<abcdl::algebra::Mat*>* train_seq_data,
               std::vector<abcdl::algebra::Mat*>* train_seq_label){

    //L(y, o) = - (1/N)(Sum y_n*log(o_n))

    real loss_value = total_loss(train_seq_data, train_seq_label);
    size_t N = 0;
    
    size_t size = train_seq_label->size();
    for(size_t i = 0; i != size; i++){
        N += train_seq_label->at(i)->rows();
    }

    return loss_value / N;
}

bool RNN::check_data(std::vector<abcdl::algebra::Mat*>* train_seq_data,
              		 std::vector<abcdl::algebra::Mat*>* train_seq_label){ 
	size_t size = train_seq_data->size();
	for(size_t i = 0; i != size; i++){
		if(train_seq_data->at(i)->rows() != train_seq_label->at(i)->rows() ||
			train_seq_data->at(i)->cols() != train_seq_label->at(i)->cols()){
			printf("train_data[%d] do not match[%d-%d][%d-%d]\n", i, train_seq_data->at(i)->rows(), train_seq_label->at(i)->rows(), train_seq_data->at(i)->cols(), train_seq_label->at(i)->cols());
			
			return false;
		}
	}
	return true;
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
    _layer = new abcdl::rnn::Layer(_hidden_dim, _bptt_truncate);

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
    return loader.write<real>(models, path, false, "RNNMODEL");
}

}//namespace rnn
}//namespace abcdl
