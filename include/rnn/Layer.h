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
	Layer(const size_t hidden_dim, const size_t bptt_truncate){
	    _hidden_dim = hidden_dim;
        _bptt_truncate = bptt_truncate;
	}

	~Layer() = default;

	void farward(const abcdl::algebra::Mat& train_seq_data,
                 const abcdl::algebra::Mat& weight,
                 const abcdl::algebra::Mat& pre_weight,
                 const abcdl::algebra::Mat& act_weight,
                 const abcdl::algebra::Mat& state,
                 abcdl::algebra::Mat& activation);

	void backward(const abcdl::algebra::Mat& train_seq_data,
				  const abcdl::algebra::Mat& train_seq_label,
                  const abcdl::algebra::Mat& weight,
                  const abcdl::algebra::Mat& pre_weight,
                  const abcdl::algebra::Mat& act_weight,
                  abcdl::algebra::Mat& derivate_weight,
                  abcdl::algebra::Mat& derivate_pre_weight,
                  abcdl::algebra::Mat& derivate_act_weight);

private:
	size_t _hidden_dim;
    size_t _bptt_truncate;
};//class Layer

}//namespace rnn
}//namespace abcdl
