/*********************************************
* Author: Jun Jiang - jiangjun4@sina.com
* Created: 2017-08-24 15:03
* Last modified: 2017-08-24 19:30
* Filename: Layer.h
* Description: CNN network layer
**********************************************/
#pragma once

#include <vector>
#include "algebra/Matrix.h"

namespace ccma{
namespace cnn{

enum Layer_type{
	INPUT = 0,
	OUTPUT
};

class Layer{
public:
    Layer(const size_t rows,
          const size_t cols,
          const size_t in_channel_size,
          const size_t out_channel_size,
		  const abcdl::cnn::Layer_type layer_type) : _rows(rows),
        _cols(cols),
        _in_channel_size(in_channel_size),
        _out_channel_size(out_channel_size),
    	_layer_type(layer_type){}

    virtual ~Layer() = default;

	inline void set_alpha(const real alpha){ _alpha = alpha; }

    virtual void forward(Layer* pre_layer) = 0;
    virtual void backward(Layer* pre_layer, Layer* back_layer) = 0;

protected:
    size_t _rows;
    size_t _cols;
    size_t _in_channel_size;
    size_t _out_channel_size;

    real _alpha = 0.1;

    std::vector<abcdl::algebra::Mat&> _weights;
    abcdl::algebra::Mat _bias;
private:
    std::vector<abcdl::algebra::Mat&> _activations;
    std::vector<abcdl::algebra::Mat&> _deltas;

	abcdl::cnn::Layer_type;
};//class Layer

class InputLayer : public Layer{
public:
    InputLayer(const size_t rows, const size_t cols) : Layer(rows, cols, 1, 1, abcdl::cnn::INPUT){}
    void forward(Layer* pre_layer);
    void backward(Layer* pre_layer, Layer* back_layer){}

    void set_x(const abcdl::algebra::Mat& x){
        CHECK(x->get_rows() == this->_rows && x->get_cols() == this->_cols){
        _x = x;
    }
private:
    abcdl::algebra::Mat _x;
};//class InputLayer

class SubSamplingLayer : public Layer{
public:
    SubSamplingLayer(size_t scale, abcdl::cnn::Pooling* pooling):Layer(0, 0, 0, 0){
        _scale = scale;
        _pooling = pooling;
    }
    ~SubSamplingLayer(){
        delete _pooling;
    }

    void forward(Layer* pre_layer);
    void backward(Layer* pre_layer, Layer* back_layer);

    inline size_t get_scale() const{ return _scale; }

private:
    size_t _scale;
	abcdl::cnn::Pooling* _pooling;
};//class SubsamplingLayer


class ConvolutionLayer : public Layer{
public:
     ConvolutionLayer(const size_t kernal_size,
					  const size_t stride,
					  const size_t out_channel_size):Layer(0, 0, 1, out_channel_size){
        _kernal_size = kernal_size;
        _stride = stride;
    }
	
    inline size_t get_stride() const { return _stride; }

    void forward(Layer* pre_layer);
    void backward(Layer* pre_layer, Layer* back_layer);

private:
    size_t _stride;
    size_t _kernal_size;
};//class ConvolutionLayer

class FullConnectionLayer : public Layer{
public:
    FullConnectionLayer(const size_t rows) : Layer(rows, 0, 0, 1){}
    
	void forward(Layer* pre_layer);
    void backward(Layer* pre_layer, Layer* back_layer);

    void set_y(const abcdl::algebra::Mat& y){
        CHECK(y.rows() == _rows);
        _y = y;
    }

private:
    abcdl::algebra::Mat _y;
};//class FullConnectionLayer

}//namespace cnn
}//namespace abcdl
