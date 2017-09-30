/***********************************************
 * Author: Jun Jiang - jiangjun4@sina.com
 * Create: 2017-09-05 15:23
 * Last modified : 2017-09-05 15:52
 * Filename      : Layer.h
 * Description   : RNN network Layer 
 **********************************************/
#pragma once

#include "algebra/Matrix.h"
#include "algebra/MatrixSet.h"
#include "algebra/MatrixHelper.h"
#include "framework/Cost.h"
#include "framework/ActivateFunc.h"

namespace abcdl{
namespace rnn{

class Layer{
public:
	Layer(const size_t hidden_dim,
		  const size_t bptt_truncate,
		  abcdl::framework::Cost* cost,
          abcdl::framework::ActivateFunc* activate_func){
	    _hidden_dim    = hidden_dim;
        _bptt_truncate = bptt_truncate;
		_cost          = cost;
        _activate_func = activate_func;
	}

	~Layer(){
        delete _cost;
        delete _activate_func;
    }

	void farward(const abcdl::algebra::Mat& train_seq_data,
                 const abcdl::algebra::Mat& weight,
                 const abcdl::algebra::Mat& pre_weight,
                 const abcdl::algebra::Mat& act_weight,
                 abcdl::algebra::Mat& state,
                 abcdl::algebra::Mat& activation);

	void backward(const abcdl::algebra::Mat& train_seq_data,
                  const abcdl::algebra::Mat& train_seq_label,
                  abcdl::algebra::Mat& weight,
                  abcdl::algebra::Mat& pre_weight,
                  abcdl::algebra::Mat& act_weight,
                  const abcdl::algebra::Mat& state,
                  const abcdl::algebra::Mat& activation,
                  abcdl::algebra::Mat& derivate_weight,
                  abcdl::algebra::Mat& derivate_pre_weight,
                  abcdl::algebra::Mat& derivate_act_weight);

private:
	size_t _hidden_dim;
    size_t _bptt_truncate;
	abcdl::framework::Cost* _cost;
    abcdl::framework::ActivateFunc* _activate_func;
	abcdl::algebra::MatrixHelper<real> helper;
};//class Layer

}//namespace rnn
}//namespace abcdl
