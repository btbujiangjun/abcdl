/***********************************************
 * Author: Jun Jiang - jiangjun4@sina.com
 * Create: 2017-09-05 15:23
 * Last modified : 2017-09-05 15:52
 * Filename      : Layer.h
 * Description   : RNN network Layer 
 **********************************************/
#pragma once

#include "algebra/Matrix.h"

namespace abcdl{
namespace rnn{

class Layer{
public:
	Layer(size_t hidden_dim,
          size_t bptt_truncate){
	    _hidden_dim = hidden_dim;
        _bptt_truncate = bptt_truncate;
	}

	~Layer() = default;

	void farward(ccma::algebra::BaseMatrixT<real>* train_seq_data,
                      ccma::algebra::BaseMatrixT<real>* weight,
                      ccma::algebra::BaseMatrixT<real>* pre_weight,
                      ccma::algebra::BaseMatrixT<real>* act_weight,
                      ccma::algebra::BaseMatrixT<real>* state,
                      ccma::algebra::BaseMatrixT<real>* activation);

	void backward(ccma::algebra::BaseMatrixT<real>* train_seq_data,
						  ccma::algebra::BaseMatrixT<real>* train_seq_label,
                          ccma::algebra::BaseMatrixT<real>* weight,
                          ccma::algebra::BaseMatrixT<real>* pre_weight,
                          ccma::algebra::BaseMatrixT<real>* act_weight,
                          ccma::algebra::BaseMatrixT<real>* derivate_weight,
                          ccma::algebra::BaseMatrixT<real>* derivate_pre_weight,
                          ccma::algebra::BaseMatrixT<real>* derivate_act_weight);

private:
	size_t _hidden_dim;
    size_t _bptt_truncate;
};//class Layer

}//namespace rnn
}//namespace abcdl
