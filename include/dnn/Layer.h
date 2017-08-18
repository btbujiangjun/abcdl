/***********************************************
 * Author: Jun Jiang - jiangjun4@sina.com
 * Create: 2017-08-16 12:01
 * Last modified : 2017-08-16 12:01
 * Filename      : Layer.h
 * Description   : 
 **********************************************/

#pragma once

#include "algebra/MatrixHelper.h"
#include "dnn/ActivateFunc.h"
#include "dnn/Cost.h"

namespace abcdl{
namespace dnn{

enum Layer_type{
    INPUT = 0,
    FULL_CONNECTION,
    OUTPUT
};


class Layer{
public:

    Layer(size_t input_dim,
          size_t output_dim,
          Layer_type layer_type){
        _input_dim  = input_dim;
        _output_dim = output_dim;
        _layer_type = layer_type;
    }

    size_t get_input_dim() const{
        return _input_dim;
    }
    size_t get_output_dim() const{
        return _output_dim;
    }
    Layer_type get_layer_type() const{
        return _layer_type;
    }
   
    abcdl::algebra::Mat& get_activate_data() const{
        return _activate_data;
    }
    
    abcdl::algebra::Mat& get_delta_weight() const{
        return _delta_weight;
    }

    abcdl::algebra::Mat& get_delta_bias() const{
        return _delta_bias;
    }

    abcdl::algebra::Mat& get_weight() const{
        return _weight;
    }

    abcdl::algebra::Mat& get_bias() const{
        return _bias;
    }

    virtual void feedward(const abcdl::algebra::Mat& mat);
    virtual void backward(Layer* next_layer);
    void update_gradient(size_t batch_size, real learning_rate){
        _weight -= _batch_weight * learning_rate / batch_size;
        _bias   -= _batch_bias * learning_rate / batch_size;
    }

protected:
    abcdl::algebra::MatrixHelper helper;

private:
    size_t _input_dim;
    size_t _output_dim;
    Layer_type _layer_type;

    abcdl::algebra::Mat _activate_data;

    abcdl::algebra::Mat _delta_weight;
    abcdl::algebra::Mat _delta_bias;
    abcdl::algebra::Mat _batch_weight;
    abcdl::algebra::Mat _batch_bias;

    abcdl::algebra::Mat _weight;
    abcdl::algebra::Mat _bias;
    
};//class Layer


class InputLayer : Layer{
public:
    InputLayer(size_t feature_dim) : Layer(feature_dim, feature_dim, INPUT){}

    void feedward(const abcdl::algebra::Mat& mat);
    void backward(Layer* next_layer);
};//class InputLayer

class FullConnLayer : Layer{
public:
    FullConnLayer(const size_t input_dim,
                  const size_t output_dim,
                  ActivateFunc* activate_func,
                  const real& mean_value = 0.0f,
                  const real& stddev = 1.0f):Layer(input_dim, output_dim, FULL_CONNECTION){
        _activate_func  = activate_func;
        _weight         = new abcdl::algebra::RandomMatrix<real>(_input_dim, _output_dim, mean_value, stddev);
        _bias           = new abcdl::algebra::Mat(1, _output_dim);
    }

    FullConnLayer(const size_t input_dim,
                  const size_t output_dim,
                  abcdl::dnn::ActivateFunc * activate_func,
                  abcdl::dnn::Cost* cost,
                  abcdl::algebra::Mat* weight,
                  abcdl::algebra::Mat* bias) : Layer(input_dim, output_dim, FULL_CONNECTION){
        _activate_func  = activate_func;
        _cost           = cost;
        _weight         = weight;
        _bias           = bias;
    }

    ~FullConnLayer(){
        delete _activate_func;
        delete _cost;
    }

    void feedward(const abcdl::algebra::Mat& mat);
    void backward(Layer* next_layer);
private:
    abcdl::dnn::ActivateFunc* _activate_func;
    abcdl::dnn::Cost* _cost;
};//class FullConnLayer

class OutputLayer : Layer{
public:
    OutputLayer(const size_t input_dim,
                const size_t output_dim,
                ActivateFunc* activate_func,
                Cost* cost) : Layer(input_dim, output_dim, FULL_CONNECTION){
        _cost           = cost;
        _activate_func  = activate_func;
//        _weight         = new abcdl::algebra::RandomMatrix<real>(_input_dim, _output_dim, mean_value, stddev);
//        _bias           = new abcdl::algebra::Mat(1, _output_dim);
    }

    ~OutputLayer(){
        delete _activate_func;
        delete _cost;
    }

    void feedward(const abcdl::algebra::Mat& mat);
    void backward(Layer* next_layer);

    void set_y(const abcdl::algebra::Mat& y){
        _y = y;
    }

private:
    abcdl::dnn:Cost* _cost;
    abcdl::algebra::Mat _y;
};//class OutputLayer

}//namespace dnn
}//namespace abcdl
