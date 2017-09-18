/***********************************************
 * Author: Jun Jiang - jiangjun4@sina.com
 * Create: 2017-09-05 15:08
 * Last modified : 2017-09-05 15:56
 * Filename      : RNN.h
 * Description   : Recurrent Neural Network 
 **********************************************/
#pragma once

#include <vector>
#include "rnn/Layer.h"
#include "utils/ModelLoader.h"

namespace abcdl{
namespace rnn{

class RNN{
public:
    RNN(const size_t feature_dim, const size_t hidden_dim){

        _feature_dim    = feature_dim;
        _hidden_dim     = hidden_dim;

        _U.reset(hidden_dim, feature_dim, 0, 1, -std::sqrt(1.0/feature_dim), std::sqrt(1.0/feature_dim));
        _W.reset(hidden_dim, hidden_dim, 0, 1, -std::sqrt(1.0/hidden_dim), std::sqrt(1.0/hidden_dim));
        _V.reset(feature_dim, hidden_dim, 0, 1, -std::sqrt(1.0/hidden_dim), std::sqrt(1.0/hidden_dim));

        _layer = new abcdl::rnn::Layer(hidden_dim, _bptt_truncate, new abcdl::framework::CrossEntropyCost());
    }

    RNN(const std::string& path){
        if(load_model(path)){
            _feature_dim    = _U.cols();
            _hidden_dim     = _U.rows();
            _path           = path;
            _layer          = new abcdl::rnn::Layer(_hidden_dim, _bptt_truncate, new abcdl::framework::CrossEntropyCost());
        }

    }
    ~RNN(){delete _layer;}

    void train(const std::vector<abcdl::algebra::Mat*>& train_seq_data,
               const std::vector<abcdl::algebra::Mat*>& train_seq_label); 

    bool load_model(const std::string& path);
    bool write_model(const std::string& path);

    void set_epoch(const size_t epoch){_epoch = epoch;}
    void set_mini_batch_size(const size_t mini_batch_size){ _mini_batch_size = mini_batch_size; }
    void set_alpha(const real alpha){_alpha = alpha;}
    void set_model_path(const std::string& path){_path = path;}
    void set_bptt_truncate(const size_t bptt_truncate){_bptt_truncate = bptt_truncate;}

private:
    void mini_batch_update(const std::vector<abcdl::algebra::Mat*>& train_seq_data,
                           const std::vector<abcdl::algebra::Mat*>& train_seq_label);

    real loss(const std::vector<abcdl::algebra::Mat*>& train_seq_data,
              const std::vector<abcdl::algebra::Mat*>& train_seq_label);

    real total_loss(const std::vector<abcdl::algebra::Mat*>& train_seq_data,
                    const std::vector<abcdl::algebra::Mat*>& train_seq_label);
    
	bool check_data(const std::vector<abcdl::algebra::Mat*>& train_seq_data,
             		const std::vector<abcdl::algebra::Mat*>& train_seq_label); 
private:
    size_t _feature_dim;
    size_t _hidden_dim;
    size_t _bptt_truncate = 4; 

    size_t _epoch = 5;
    size_t _mini_batch_size = 10;
    real _alpha = 0.1;

    std::string _path = "./model/rnn.model";
    abcdl::utils::ModelLoader loader;

    abcdl::algebra::RandomMatrix<real> _U;
    abcdl::algebra::RandomMatrix<real> _W;
    abcdl::algebra::RandomMatrix<real> _V;

    abcdl::rnn::Layer* _layer;
};//class RNN 

}//namespace rnn
}//namespace abcdl
