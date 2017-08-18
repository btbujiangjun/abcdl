/***********************************************
 * Author: Jun Jiang - jiangjun4@sina.com
 * Create: 2017-08-16 16:11
 * Last modified : 2017-08-16 16:11
 * Filename      : Layer.h
 * Description   : 
 **********************************************/
#include "dnn/Layer.h"
#include "utils/Log.h"

namespace abcdl{
namespace dnn{

void InputLayer::feedward(const abcdl::algebra::Mat& mat){
    CHECK(mat.get_rows() != _input_dim);
    _activate_data  = mat;
}
void InputLayer::backward(Layer* next_layer){
}

void FullConnLayer::feedward(const abcdl::algebra::Mat& mat){
    //activate_func(x * w + b)
    _activate_func->activate(_activate_data, mat * _weight + _bias);
}
void FullConnLayer::backward(Layer* next_layer){
    if(next_layer->get_layer_type() == abcdl::dnn::Output){
        _delta_bias = next_layer->get_bias();
    }else{
        //δ_l = ( (w_l+1).T * δ_l+1 ) * Derivative(z_l)
        abcdl::algebra::Mat activate_derivative;
        _activate_func->derivative(&activate_derivative, &_activate_data);
        _delta_bias   = next_layer->get_weight().Ts() * next_layer->get_bias() * activate_derivative;
    }
    //_weight = activate_mat * bias
    /*
     * Derivative(Cw) = a_in * δ_out
     * a_in = a_l-1, δ_out = mat
     * activations include input layer, so l-1 is i.
     */
    _delta_weight   += _activate_data * _bias;

    _batch_bias     += _delta_bias;
    _batch_weight   += _delta_weight;
}

void OutputLayer::feedward(const abcdl::algebra::Mat& mat){
    //activate_func(x * w + b)
    _activate_func->activate(_activate_data, mat * _weight + _bias);
}
void OutputLayer::backward(Layer* next_layer){
    /*
     * backpropagation
     * L layer(last layer) Error
     * Error δL = cost->delta
     */
    _cost->delta(_bias, _activate_data, _y);
}

}//namespace dnn
}//namespace abcdl
