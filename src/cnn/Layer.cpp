/*********************************************
* Author: Jun Jiang - jiangjun4@sina.com
* Created: 2017-05-22 16:47
* Last modified: 2017-05-22 16:47
* Filename: CNN.cpp
* Description: convolutional network 
**********************************************/
#include <string.h>
#include "cnn/Layer.h"

namespace abcdl{
namespace cnn{

void SubSamplingLayer::initialize(Layer* pre_layer){
    CHECK(pre_layer->get_rows() % _scale == 0 && pre_layer->get_cols() % _scale == 0);
	
    this->_rows = pre_layer->get_rows() / _scale;
    this->_cols = pre_layer->get_cols() / _scale;

    //can't change out_channel_size.
    this->_in_channel_size = this->_out_channel_size = pre_layer->get_out_channel_size();

    this->_deltas.reserve(this->_out_channel_size);
    for(size_t i = 0; i != this->_out_channel_size; i++){
        this->_deltas.push_back(new abcdl::algebra::Mat());
        this->_activations.push_back(new abcdl::algebra::Mat());
    }
}
void SubSamplingLayer::forward(Layer* pre_layer){
	abcdl::algebra::Mat activation;
    for(size_t i = 0; i != this->_out_channel_size; i++){
        _pooling->pool(activation, pre_layer->get_activation(i), this->_rows, this->_cols, this->_scale);
        this->set_activation(i, activation);
    }
}
void SubSamplingLayer::backward(Layer* pre_layer, Layer* back_layer){
    if(back_layer->get_layer_type() == abcdl::framework::OUTPUT){
        size_t size = this->_rows * this->_cols;
        real*  data = back_layer->get_delta(0).data();
        abcdl::algebra::Mat delta(this->_rows, this->_cols);
        
        //recover multiply mat
        for(size_t i = 0; i != this->_out_channel_size; i++){
            memcpy(delta.data(), &data[i * size], sizeof(real) * size);
            this->set_delta(i, delta);
        }
    }else if(back_layer->get_layer_type() == abcdl::framework::CONVOLUTION){
        size_t stride = ((ConvolutionLayer*)back_layer)->get_stride();
        for(size_t i = 0 ; i != this->_out_channel_size; i++){
            abcdl::algebra::Mat delta;
            abcdl::algebra::Mat sub_delta;
            for(size_t j = 0; j != back_layer->get_out_channel_size(); j++){
                _helper.convn(sub_delta, back_layer->get_delta(j), back_layer->get_weight(i, j), stride, abcdl::algebra::FULL);
                delta += sub_delta;
            }
            this->set_delta(i, delta);
        }
    }
}

void ConvolutionLayer::initialize(Layer* pre_layer){
    size_t pre_rows = pre_layer->get_rows();
    size_t pre_cols = pre_layer->get_cols();

    CHECK(pre_rows > _kernal_size && pre_cols > _kernal_size);

    this->_rows = (pre_rows - _kernal_size) % _stride == 0 ? (pre_rows - _kernal_size) / _stride + 1 : (pre_rows - _kernal_size) / _stride + 2;
    this->_cols = (pre_cols - _kernal_size) % _stride == 0 ? (pre_cols - _kernal_size) / _stride + 1 : (pre_cols - _kernal_size) / _stride + 2;

    this->_in_channel_size = pre_layer->get_out_channel_size();

    size_t size = this->_in_channel_size * this->_out_channel_size;
    this->_batch_weights.reserve(size);
    this->_delta_weights.reserve(size);
	this->_weights.reserve(size);
	for(size_t i = 0; i != size; i++){
        this->_weights.push_back(new abcdl::algebra::RandomMatrix<real>(_kernal_size, _kernal_size, 0.0, 0.5));
        this->_batch_weights.push_back(new abcdl::algebra::Mat(_kernal_size, _kernal_size));
        this->_delta_weights.push_back(new abcdl::algebra::Mat(_kernal_size, _kernal_size));
	}

    //all channels shared the same bias of current layer.
	this->_bias->reset(0, this->_out_channel_size, 1);
	this->_delta_bias->reset(0, this->_out_channel_size, 1);
	this->_batch_bias->reset(0, this->_out_channel_size, 1);
    
    for(size_t i = 0; i != this->_out_channel_size; i++){
        this->_deltas.push_back(new abcdl::algebra::Mat());
        this->_activations.push_back(new abcdl::algebra::Mat());
    }
}

void ConvolutionLayer::forward(Layer* pre_layer){
    for(size_t i = 0; i != this->_out_channel_size; i++){
        abcdl::algebra::Mat activation;
        abcdl::algebra::Mat pre_activation;
        for(size_t j = 0; j != pre_layer->get_out_channel_size(); j++){
            pre_activation = pre_layer->get_activation(j);
            pre_activation.convn(this->get_weight(j, i), _stride, abcdl::algebra::VALID);
            activation += pre_activation;//sum all channels of pre_layer
        }
        //add shared bias of channel in current layer.
        activation += this->_bias->get_data(i, 0);

        _activate_func->activate(activation, activation);
        this->set_activation(i, activation);
    }
}

void ConvolutionLayer::backward(Layer* pre_layer, Layer* back_layer){
    if(back_layer->get_layer_type() == abcdl::framework::OUTPUT){
        size_t size = this->_rows * this->_cols;
        real*  data = back_layer->get_delta(0).data();
        abcdl::algebra::Mat delta(this->_rows, this->_cols);
        
        //recover multiply mat
        for(size_t i = 0; i != this->_out_channel_size; i++){
            memcpy(delta.data(), &data[i * size], sizeof(real) * size);
            this->set_delta(i, delta);
        }
    }else if(back_layer->get_layer_type() == abcdl::framework::SUBSAMPLING){
        SubSamplingLayer* sub_layer = (SubSamplingLayer*)back_layer;
        size_t scale = sub_layer->get_scale();
        abcdl::algebra::Mat delta;
        abcdl::algebra::Mat back_delta;
        for(size_t i = 0; i != this->_out_channel_size; i++){
            _activate_func->derivative(delta, this->get_activation(i));
            back_delta = back_layer->get_delta(i);
            //subsampling layer reduced matrix dim, so recover it by expand function
            back_delta.expand(scale, scale);
            //back layer error sharing
            back_delta /= scale * scale;
            //delta_l = derivative_sigmoid * delta_l+1(recover dim)
            delta *= back_delta;

            this->set_delta(i, delta);
	    }
    }

    for(size_t i = 0; i != this->_out_channel_size; i++){
        abcdl::algebra::Mat weight;
        for(size_t j = 0; j != pre_layer->get_out_channel_size(); j++){
            _helper.convn(weight, pre_layer->get_activation(j), this->get_delta(i), _stride, abcdl::algebra::VALID);
            this->set_delta_weight(j, i, weight);
            this->set_batch_weight(j, i, weight);
        }

        this->_delta_bias->set_data(this->get_delta(i).sum(), i, 0);
    }
    this->_batch_bias->operator+=(*this->_delta_bias);
}

void OutputLayer::initialize(Layer* pre_layer){
    this->_cols = pre_layer->get_rows() * pre_layer->get_cols() * pre_layer->get_out_channel_size();
    this->_in_channel_size = pre_layer->get_out_channel_size();

    this->_weights.push_back(new abcdl::algebra::RandomMatrix<real>(this->_rows, this->_cols, 0.0, 0.5));
    this->_bias = new abcdl::algebra::Mat(0.0, _rows, 1);
    this->_activations.push_back(new abcdl::algebra::Mat());
    this->_deltas.push_back(new abcdl::algebra::Mat());

    this->_delta_weights.push_back(new abcdl::algebra::Mat(this->_rows, this->_cols));
    this->_delta_bias = new abcdl::algebra::Mat(0.0, _rows, 1);
    this->_batch_weights.push_back(new abcdl::algebra::Mat(this->_rows, this->_cols));
    this->_batch_bias = new abcdl::algebra::Mat(0.0, _rows, 1);
}
void OutputLayer::forward(Layer* pre_layer){
    //concatenate pre_layer's all channel mat into array
	size_t size = pre_layer->get_rows() * pre_layer->get_cols() * this->_in_channel_size;
    real* data = new real[size];
    size_t idx = 0;
    for(size_t i = 0; i != this->_in_channel_size; i++){
        size_t activation_size = pre_layer->get_activation(i).get_size();
        memcpy(&data[idx], pre_layer->get_activation(i).data(), sizeof(real) * activation_size);
        idx += activation_size;
    }

    _pre_activation_array.set_shallow_data(data, size, 1);

    auto activation = _helper.dot(this->get_weight(0, 0), _pre_activation_array) + (*this->_bias);
    _activate_func->activate(activation, activation);
    this->set_activation(0, activation);
}

void OutputLayer::backward(Layer* pre_layer, Layer* back_layer){
    abcdl::algebra::Mat error;
    _cost->delta(error, this->get_activation(0), _y);

    _loss = (error * error).sum() / 2;

    //error * derivate_of_output
    abcdl::algebra::Mat derivative_output;
    _activate_func->derivative(derivative_output, this->get_activation(0));
    derivative_output *= error;

    //calc delta: weight.T * derivate_output
    this->set_delta(0, _helper.dot(this->get_weight(0, 0).Ts(), derivative_output));

    //if pre_layer is ConvolutionLayer, has sigmoid function
    if(pre_layer->get_layer_type() == abcdl::framework::CONVOLUTION){
        abcdl::algebra::Mat mat;
        _activate_func->derivative(mat, _pre_activation_array);
        this->get_delta(0) *= mat;
    }

    //derivate_weight = derivate_output * _pre_activation_array.T
    //derivate_bias = derivate_output
    auto delta_weight = _helper.dot(derivative_output, _pre_activation_array.Ts());
    this->set_delta_weight(0, 0, delta_weight);
    (*this->_delta_bias) = derivative_output;

    this->set_batch_weight(0, 0, this->get_delta_weight(0, 0)); 
    (*this->_batch_bias) += derivative_output;
}

}//namespace cnn
}//namespace abcdl
