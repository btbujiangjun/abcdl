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

void InputLayer::forward(const abcdl::algebra::Mat& mat){
    CHECK(mat.cols() == _input_dim);
    this->_activate_data  = mat;
}
void InputLayer::backward(Layer* pre_layer, Layer* next_layer){
}

void FullConnLayer::forward(const abcdl::algebra::Mat& mat){
    //activate_func(x * w + b)
    _activate_func->activate(this->_activate_data, helper.dot(mat, this->_weight) + this->_bias);
}
void FullConnLayer::backward(Layer* pre_layer, Layer* next_layer){
    //δ_l = ( (w_l+1).T * δ_l+1 ) * Derivative(a_l)
    abcdl::algebra::Mat activate_derivative;
    _activate_func->derivative(activate_derivative, this->_activate_data);
    _delta_bias   = helper.dot(helper.dot(next_layer->get_weight().Ts(), next_layer->get_bias()), activate_derivative);
    
    //_weight = activate_mat * bias
    /*
     * Derivative(Cw) = a_in * δ_out
     * a_in = a_l-1, δ_out = mat
     * activations include input layer, so l-1 is i.
     */
    this->_delta_weight   = helper.dot(this->_activate_data, this->_delta_bias);

    this->_batch_bias     += this->_delta_bias;
    this->_batch_weight   += this->_delta_weight;
}

void OutputLayer::forward(const abcdl::algebra::Mat& mat){
    //activate_func(x * w + b)
    _activate_func->activate(this->_activate_data, helper.dot(mat, this->_weight) + this->_bias);
}
void OutputLayer::backward(Layer* pre_layer, Layer* next_layer){
    /*
     * L layer(last layer) Error
     * Error δL = cost->delta
     */
    _cost->delta(this->_delta_bias, this->_activate_data, this->_y);
    /*
     * Derivative(Cw) = a_in * δ_out
     * a_in = a_L-1, δ_out = delta
     */
    this->_delta_weight   = helper.dot(pre_layer->get_activate_data(), this->_delta_bias);

    this->_batch_weight   += this->_delta_weight;
    this->_batch_bias     += this->_delta_bias;

}

}//namespace dnn
}//namespace abcdl