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

void InputLayer::set_x(const abcdl::algebra::Mat& mat){
    CHECK(mat.cols() == _input_dim);
    this->_activate_data  = mat;
}

void FullConnLayer::forward(Layer* pre_layer){
    //activate_func(x * w + b)
    _activate_func->activate(this->_activate_data, helper.dot(pre_layer->get_activate_data(), this->_weight) + this->_bias);
    this->_activate_data.display("^");
}
void FullConnLayer::backward(Layer* pre_layer, Layer* next_layer){
    //δ_l = ( (w_l+1).T .* δ_l+1 ) * Derivative(a_l)
    abcdl::algebra::Mat activate_derivative;
    _activate_func->derivative(activate_derivative, this->_activate_data);
    _delta_bias   = helper.dot(next_layer->get_delta_bias(), next_layer->get_weight().Ts()) * activate_derivative;
    
    //_weight = activate_mat * bias
    /*
     * Derivative(Cw) = a_in * δ_out
     * a_in = a_l-1, δ_out = mat
     * activations include input layer, so l-1 is i.
     */
    this->_delta_weight   = helper.dot(pre_layer->get_activate_data().Ts(), this->_delta_bias);


    this->_batch_bias     += this->_delta_bias;
    this->_batch_weight   += this->_delta_weight;
}

void OutputLayer::forward(Layer* pre_layer){
    //activate_func(x * w + b)
    _activate_func->activate(this->_activate_data, helper.dot(pre_layer->get_activate_data(), this->_weight) + this->_bias);
    this->_activate_data.display("^");
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
    this->_delta_weight   = helper.dot(pre_layer->get_activate_data().Ts(), this->_delta_bias);

    this->_batch_weight   += this->_delta_weight;
    this->_batch_bias     += this->_delta_bias;
}

}//namespace dnn
}//namespace abcdl
