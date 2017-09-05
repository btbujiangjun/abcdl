/***********************************************
 * Author: Jun Jiang - jiangjun4@sina.com
 * Create: 2017-09-05 15:08
 * Last modified : 2017-09-05 15:56
 * Filename      : RNN.h
 * Description   : Recurrent Neural Network 
 **********************************************/
#pragma once

#include <vector>
#include <thread>
#include "algorithm/rnn/Layer.h"
#include "utils/ModelLoader.h"

namespace abcdl{
namespace rnn{

class RNN{
public:
    RNN(size_t feature_dim,
        size_t hidden_dim,
        const std::string path = "",
        size_t bptt_truncate = 4){

        _feature_dim    = feature_dim;
        _hidden_dim     = hidden_dim;
        _bptt_truncate  = bptt_truncate;
        _path           = path;

        _U.reset(hidden_dim, feature_dim, 0, 1, -std::sqrt(1.0/feature_dim), std::sqrt(1.0/feature_dim));
        _W.reset(hidden_dim, hidden_dim, 0, 1, -std::sqrt(1.0/hidden_dim), std::sqrt(1.0/hidden_dim));
        _V.reset(feature_dim, hidden_dim, 0, 1, -std::sqrt(1.0/hidden_dim), std::sqrt(1.0/hidden_dim));

        _layer = new abcdl::rnn::Layer(hidden_dim, bptt_truncate);
    }

    RNN(const std::string& path, const size_t bptt_truncate = 4){
        if(load_model(path)){
            _feature_dim    = _U.cols();
            _hidden_dim     = _U.rows();
            _bptt_truncate  = bptt_truncate;
            _path           = path;
            _layer          = new abcdl::rnn::Layer(_hidden_dim, _bptt_truncate);
        }

    }
    ~RNN(){delete _layer;}

    void sgd(std::vector<abcdl::algebra::BaseMatrixT<real>*>* train_seq_data,
             std::vector<abcdl::algebra::BaseMatrixT<real>*>* train_seq_label, 
             const size_t epoch = 5,
             const size_t mini_batch_size = 1,
             const real alpha = 0.1);

    bool load_model(const std::string& path);
    bool write_model(const std::string& path);

    void set_epoch(const size_t epoch) const {_epoch = epoch;}
    void set_mini_batch_size(const size_t mini_batch_size) const { _mini_batch_size = mini_batch_size; }
    void set_alpha(const real alpha) const {_alpha = alpha;}

private:
    void mini_batch_update(std::vector<abcdl::algebra::BaseMatrixT<real>*> train_seq_data,
                           std::vector<abcdl::algebra::BaseMatrixT<real>*> train_seq_label, 
			  	           const real alpha,
                           const bool debug,
				           const int j);

    real loss(std::vector<abcdl::algebra::BaseMatrixT<real>*>* train_seq_data,
              std::vector<abcdl::algebra::BaseMatrixT<real>*>* train_seq_label);

    real total_loss(std::vector<abcdl::algebra::BaseMatrixT<real>*>* train_seq_data,
                    std::vector<abcdl::algebra::BaseMatrixT<real>*>* train_seq_label);
    
	bool check_data(std::vector<abcdl::algebra::BaseMatrixT<real>*>* train_seq_data,
             		std::vector<abcdl::algebra::BaseMatrixT<real>*>* train_seq_label); 
private:
    size_t _feature_dim;
    size_t _hidden_dim;
    size_t _bptt_truncate; 

    size_t _epoch = 5;
    size_t _mini_batch_size = 1;
    real _alpha = 0.1;

    std::string _path;
    abcdl::utils::ModelLoader loader;

    abcdl::algebra::Mat _U;
    abcdl::algebra::Mat _W;
    abcdl::algebra::Mat _V;

    Layer* _layer;
};//class RNN 

}//namespace rnn
}//namespace abcdl

#endif
