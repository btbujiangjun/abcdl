/***********************************************
 * Author: Jun Jiang - jiangjun4@sina.com
 * Create: 2017-08-16 12:01
 * Last modified : 2017-09-01 10:29
 * Filename      : Layer.h
 * Description   : Layer of FNN 
 **********************************************/

#pragma once

#include "framework/Layer.h"
#include "framework/Cost.h"
#include "framework/ActivateFunc.h"
#include "algebra/MatrixHelper.h"

namespace abcdl{
namespace fnn{

class Layer{
public:
    Layer(const size_t input_dim,
          const size_t output_dim,
          const abcdl::framework::Layer_type layer_type){
        _input_dim  = input_dim;
        _output_dim = output_dim;
        _layer_type = layer_type;
    }
    virtual ~Layer() = default;

    virtual void forward(Layer* pre_layer) = 0;
    virtual void backward(Layer* pre_layer, Layer* next_layer) = 0;
    void update_gradient(const size_t batch_size,
                         const real learning_rate,
                         const real lamda){
        real weight_decay = 1.0 - learning_rate * (lamda / batch_size);
        _weight *= weight_decay;
        _weight -= _batch_weight * learning_rate / batch_size;
        _bias   -= _batch_bias * learning_rate / batch_size;
        
		_batch_weight.reset(0);
        _batch_bias.reset(0);
    }

    size_t get_input_dim() const{ return _input_dim; }
    size_t get_output_dim() const{ return _output_dim; }
    abcdl::framework::Layer_type get_layer_type() const{return _layer_type;}

    abcdl::algebra::Mat& get_weight(){  return _weight; }
    abcdl::algebra::Mat& get_bias(){ return _bias; }
    abcdl::algebra::Mat& get_activate_data(){ return _activate_data; }
    
    abcdl::algebra::Mat& get_delta_weight(){ return _delta_weight; }
    abcdl::algebra::Mat& get_delta_bias(){ return _delta_bias; }

protected:
    size_t _input_dim;
    size_t _output_dim;
    abcdl::framework::Layer_type _layer_type;
    abcdl::algebra::MatrixHelper<real> _helper;

    abcdl::algebra::RandomMatrix<real> _weight;
    abcdl::algebra::RandomMatrix<real> _bias;
    abcdl::algebra::Mat _activate_data;

    abcdl::algebra::Mat _delta_weight;
    abcdl::algebra::Mat _delta_bias;
    abcdl::algebra::Mat _batch_weight;
    abcdl::algebra::Mat _batch_bias;
};//class Layer


class InputLayer : public Layer{
public:
    InputLayer(const size_t feature_dim) : Layer(feature_dim, feature_dim, abcdl::framework::INPUT){}

    void forward(Layer* pre_layer){}
    void backward(Layer* pre_layer, Layer* next_layer){}
    
	void set_x(const abcdl::algebra::Mat& mat);
};//class InputLayer

class FullConnLayer : public Layer{
public:
    FullConnLayer(const size_t input_dim,
                  const size_t output_dim,
                  abcdl::framework::ActivateFunc* activate_func,
                  const real& mean_value = 0.0f,
                  const real& stddev = 0.5f) : Layer(input_dim, output_dim, abcdl::framework::FULL_CONN){
        _activate_func  = activate_func;

        this->_weight.reset(_input_dim, _output_dim, mean_value, stddev);
        this->_bias.reset(1, _output_dim, mean_value, stddev);
    }

    FullConnLayer(const size_t input_dim,
                  const size_t output_dim,
                  abcdl::framework::ActivateFunc * activate_func,
                  const abcdl::algebra::Mat& weight,
                  const abcdl::algebra::Mat& bias) : Layer(input_dim, output_dim, abcdl::framework::FULL_CONN){
        _activate_func  = activate_func;
        this->_weight   = weight;
        this->_bias     = bias;
    }

    ~FullConnLayer(){
        delete _activate_func;
    }

    void forward(Layer* pre_layer);
    void backward(Layer* pre_layer, Layer* next_layer);
private:
    abcdl::framework::ActivateFunc* _activate_func;
};//class FullConnLayer

class OutputLayer : public Layer{
public:
    OutputLayer(const size_t input_dim,
                const size_t output_dim,
                abcdl::framework::ActivateFunc* activate_func,
                abcdl::framework::Cost* cost,
                const real& mean_value = 0.0f,
                const real& stddev = 0.5f) : Layer(input_dim, output_dim, abcdl::framework::OUTPUT){
        _cost           = cost;
        _activate_func  = activate_func;

        this->_weight.reset(_input_dim, _output_dim, mean_value, stddev);
        this->_bias.reset(1, _output_dim, mean_value, stddev);
    }

    OutputLayer(const size_t input_dim,
                const size_t output_dim,
                abcdl::framework::ActivateFunc* activate_func,
                abcdl::framework::Cost* cost,
                const abcdl::algebra::Mat& weight,
                const abcdl::algebra::Mat& bias,
                const real& mean_value = 0.0f,
                const real& stddev = 0.5f) : Layer(input_dim, output_dim, abcdl::framework::OUTPUT){
        _cost           = cost;
        _activate_func  = activate_func;
        this->_weight   = weight;
        this->_bias     = bias;;
    }

    ~OutputLayer(){
        delete _activate_func;
        delete _cost;
    }

    void forward(Layer* pre_layer);
    void backward(Layer* pre_layer, Layer* next_layer);

    void set_y(const abcdl::algebra::Mat& y){
        _y = y;
    }

private:
    abcdl::framework::ActivateFunc* _activate_func;
    abcdl::framework::Cost* _cost;
    abcdl::algebra::Mat _y;
};//class OutputLayer

}//namespace fnn
}//namespace abcdl