/*********************************************
* Author: Jun Jiang - jiangjun4@sina.com
* Created: 2017-05-22 16:47
* Last modified: 2017-05-22 16:47
* Filename: CNN.cpp
* Description: convolutional network 
**********************************************/
#include <math.h>
#include "cnn/Layer.h"

namespace abcdl{
namespace cnn{

void SubSamplingLayer::initialize(Layer* pre_layer){
    CHECK(pre_layer->get_rows() % _scale == 0 && pre_layer->get_cols() % _scale == 0);
	
    this->_rows = pre_layer->get_rows() / _scale;
    this->_cols = pre_layer->get_cols() / _scale;

    //can't change out_channel_size.
    this->_in_channel_size = this->_out_channel_size = pre_layer->get_out_channel_size();

    //pre_layer, all channels share the same bias and initial value is zero. without pooling weight.
    this->_bias.reset(0.0, this->_out_channel_size, 1);
    this->_deltas.reserve(this->_out_channel_size);
}
void SubSamplingLayer::forward(Layer* pre_layer){
    // in_channel size equal out_channel size to subsampling layer.
	abcdl::algebra::Mat activation;
    for(size_t i = 0; i != this->_in_channel_size; i++){
        _pooling->pool(activation, pre_layer->get_activation(i), this->_rows, this->_cols, this->_scale);
        this->set_activation(i, activation);
    }
}
void SubSamplingLayer::backward(Layer* pre_layer, Layer* back_layer){
    if(back_layer->get_layer_type() == abcdl::algebra::OUTPUT){
        size_t size = this->_rows * this->_cols;
        real*  data = back_layer->get_delta(0).data();
        real*  d    = new real[size];
        abcdl::algebra::Mat delta;
        
        //recover to multiply mat
        for(size_t i = 0; i != this->_out_channel_size; i++){
            memcpy(d, &data[i * size], sizeof(real) * size);
            mat.set_shallow_data(d, this->_rows, this->_cols);
            this->set_delta(i, delta);
        }
    }else if(back_layer->get_layer_type() == abcdl::algebra::CONVOLUTION){
        size_t stride = ((ConvolutionLayer*)back_layer)->get_stride();
        for(size_t i = 0 ; i != this->_out_channel_size; i++){
            abcdl::algebra::Mat delta;
            abcdl::algebra::Mat sub_delta;
            for(size_t j = 0; j != back_layer->get_out_channel_size(); j++){
                mh.convn(sub_delta, back_layer->get_delta(j), back_layer->get_weight(i, j), stride, abcdl::algebra::FULL);
                delta += sub_delta;
            }
            this->set_delta(i, delta);
        }
    }
}

void ConvolutionLayer::initialize(Layer* pre_layer){
    size_t pre_rows = pre_layer->get_rows();
    size_t pre_cols = pre_layer->get_cols();

    CHECK(pre_rows > _kernal_size && pre_cols < _kernal_size);

    this->_rows = (pre_rows - _kernal_size) % _stride == 0 ? (pre_rows - _kernal_size) / _stride + 1 : (pre_rows - _kernal_size) / _stride + 2;
    this->_cols = (pre_cols - _kernal_size) % _stride == 0 ? (pre_cols - _kernal_size) / _stride + 1 : (pre_cols - _kernal_size) / _stride + 2;

    this->_in_channel_size = pre_layer->get_out_channel_size();
	
    this->_batch_weights.reserve(this->_in_channel_size * this->_out_channel_size);
    this->_delta_weights.reserve(this->_in_channel_size * this->_out_channel_size);
	this->_weights.reserve(this->_in_channel_size * this->_out_channel_size);
	for(auto& weight : _weights){
		weight.reset(_kernal_size, _kernal_size, 0.0, 0.5);
        weight.display("|");
	}

    //all channels shared the same bias of current layer.
	this->_bias.reset(0.0, this->_out_channel_size, 1);
}

void ConvolutionLayer::forward(Layer* pre_layer){
    for(size_t i = 0; i != this->_out_channel_size; i++){
        abcdl::algebra::Mat activation;
        for(size_t j = 0; j != pre_layer->get_out_channel_size(); j++){
            auto pre_activation = pre_layer->get_activation(j);
            pre_activation.convn(this->get_weight(j, i), _stride, abcdl::algebra::VALID);
            activation += pre_activation;//sum all channels of pre_layer
        }
        //add shared bias of channel in current layer.
        activation += this->_bias().get_data(i, 0);

        //if sigmoid activative function.
        activation.sigmoid();
        this->set_activation(i, activation);
    }
}

void ConvolutionLayer::backward(Layer* pre_layer, Layer* back_layer){
    if(back_layer->get_layer_type() == abcdl::algebra::OUTPUT){
        size_t size = this->_rows * this->_cols;
        real*  data = back_layer->get_delta(0).data();
        real*  d    = new real[size];
        abcdl::algebra::Mat delta;
        
        //recover to multiply mat
        for(size_t i = 0; i != this->_out_channel_size; i++){
            memcpy(d, &data[i * size], sizeof(real) * size);
            mat.set_shallow_data(d, this->_rows, this->_cols);
            this->set_delta(i, delta);
        }
    }else if(back_layer->get_layer_type() == abcdl::algebra::SUBSAMPLING){
        SubSamplingLayer* sub_layer = (SubSamplingLayer*)back_layer;
        size_t scale = sub_layer->get_scale();

        for(size_t i = 0; i != this->_out_channel_size; i++){
            abcdl::algebra::Mat delta;
            mh.sigmoid_derivative(delta, this->get_activation(i))

            auto back_delta = back_layer->get_delta(i);

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
        for(size_t j = 0; j != pre_layer->get_out_channel_size(); j++){
            abcdl::algebra::Mat weight;
            mh.convn(weight, pre_layer->get_activation(j), this->get_delta(i), _stride, abcdl::algebra::VALID);
            this->set_delta_weight(j, i, weight);
            this->get_batch_weight(j, i) += weight;
        }

        this->_delta_bias.set_data(this->get_delta(i).sum(), i, 0);
    }
    this->_batch_bias += this->_delta_bias;
}

void OutputLayer::initialize(Layer* pre_layer){
	//concat all vector to a array 
    this->_cols = pre_layer->rows() * pre_layer->cols() * pre_layer->get_out_channel_size();
    this->_in_channel_size = pre_layer->get_out_channel_size();

    this->_batch_weights.reserve(1);
    this->_delta_weights.reserve(1);
    this->_weights.reserve(1);
    this->_weights[0] = abcdl::algebra::RandomMatrix<real>(this->_rows, this->_cols, 0.0, 0.5);
    this->_bias.reset(0.0, _rows, 1);
}
void OutputLayer::forward(Layer* pre_layer){
    //concatenate pre_layer's all channel mat into array
	size_t size = pre_layer->get_rows() * pre_layer->get_cols();
    T* data = new T[size];
    size_t idx = 0;
    for(size_t i = 0; i != this->_in_channel_size; i++){
        size_t activation_size = pre_layer->get_activation(i).get_size();
        memcpy(&data[idx], pre_layer.get_activation(i).data(), sizeof(T) * activation_size);
        idx += activation_size;
    }

    _pre_activation_array.set_shallow_data(data, size, 1);

    auto activation = mh.dot(this->get_weight(0, 0), _pre_activation_array) + this->_bias;

    //if sigmoid activative function
    activation->sigmoid();
    this->set_activation(0, activation);
}

void OutputLayer::backward(Layer* pre_layer, Layer* back_layer){

    auto error = this->get_activation(0) - _y;
    _loss = (error * error).sum() / 2;

    //error * derivate_of_output
    abcdl::algebra::Mat derivative_output;
    mh.sigmoid_derivative(derivative_output, this->get_activation(0));
    derivative_output *= error;

    //calc delta: weight.T * derivate_output
    this->_deltas[0] = mh.dot(this->_weights[0].Ts(), derivative_output);

    //if pre_layer is ConvolutionLayer, has sigmoid function
    if(pre_layer->get_layer_type() == abcdl::algebra::CONVOLUTION){
        abcdl::algebra::Mat mat;
        mh.sigmoid_derivative(mat, _pre_activation_array);
        this->_deltas[0] *= mat;
    }

    //derivate_weight = derivate_output * _pre_activation_array.T
    //derivate_bias = derviate_output
    this->_delta_weights[0] =  mh.dot(derivative_output, _pre_activation_array.Ts());
    this->_delta_bias       = derivative_output;

    this->_batch_weight[0] += derivate_weight; 
    this->_batch_bias += derivative_output;
}

}//namespace cnn
}//namespace abcdl
