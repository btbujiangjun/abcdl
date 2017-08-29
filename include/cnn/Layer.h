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
#include "algebra/MatrixHelper.h"

namespace abcdl{
namespace cnn{

enum Layer_type{
	INPUT = 0,
	SUBSAMPLING,
	CONVOLUTION,
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

	size_t get_rows() const { return _rows; }
	size_t get_cols() const { return _cols; }

	inline size_t get_out_channel_size() { return _out_channel_size; }

	inline void set_alpha(const real alpha){ _alpha = alpha; }

	virtual void initialize(Layer* pre_layer) = 0;
    virtual void forward(Layer* pre_layer) = 0;
    virtual void backward(Layer* pre_layer, Layer* back_layer) = 0;

    void update_gradient(const size_t batch_size, const real alpha){
        for(size_t i = 0; i != _weights.size(); i++){
            _weights[i] -= _batch_weights[i] * (alpha / batch_size);
        }
        _bias -= _batch_bias *(algebra / batch_size);

        for(auto& weight : _batch_weights){
            weight.reset();
        }
        _bias.reset();
    }

protected:
    /*
	inline void set_weight(const size_t in_channel_id,
						   const size_t out_channel_id,
						   const abcdl::algebra::RandomMatrix<real>& weight){
		CHECK(in_channel_id < _in_channel_size && out_channel_id < _out_channel_size);

		if(_weights.size() != this->_out_channel_size * this->_in_channel_size){
			_weights.reserve(this->_out_channel_size * this->_in_channel_size);
		}
		
		_weights[in_channel_id * this->_out_channel_size + out_channel_id] = weight;
	}
    */
	inline abcdl::algebra::RandomMatrix<real>& get_weight(const size_t in_channel_id, const size_t out_channel_id){ 
		return _weights[in_channel_id * this->_out_channel_size + out_channel_id];
	}

	inline void set_delta_weight(const size_t in_channel_id,
						         const size_t out_channel_id,
                                 const abcdl::algebra::RandomMatrix<real>& weight){
		CHECK(in_channel_id < _in_channel_size && out_channel_id < _out_channel_size);

		if(_delta_weights.size() != this->_out_channel_size * this->_in_channel_size){
			_delta_weights.reserve(this->_out_channel_size * this->_in_channel_size);
		}
		
		_delta_weights[in_channel_id * this->_out_channel_size + out_channel_id] = weight;
	}

	abcdl::algebra::RandomMatrix<real>& get_delta_weight(const size_t in_channel_id, const size_t out_channel_id){ 
		return _delta_weights[in_channel_id * this->_out_channel_size + out_channel_id];
	}
/*
	inline void set_batch_weight(const size_t in_channel_id,
						         const size_t out_channel_id,
       						     const abcdl::algebra::RandomMatrix<real>& weight){
		CHECK(in_channel_id < _in_channel_size && out_channel_id < _out_channel_size);

		if(_batch_weights.size() != this->_out_channel_size * this->_in_channel_size){
			_batch_weights.reserve(this->_out_channel_size * this->_in_channel_size);
		}
		
		_batch_weights[in_channel_id * this->_out_channel_size + out_channel_id] = weight;
	}
*/

    void set_delta(size_t id, const abcdl::algebra::Mat& delta){
        CHECK(id < _deltas.size());
        _deltas[id] = delta;
    }
    abcdl::algebra::Mat& get_delta(const size_t id) const {
        CHECK(id < _deltas.size());
        return _deltas[id];
    }

	void set_activations(const size_t id, const abcdl::algebra::Mat& activation){
		CHECK(id < _in_channel_size);
		CHECK(activation.rows() == this->_rows && activation.cols() == this->_cols);

		if(_activations.size() != this->_in_channel_size){
			_activations.reserve(this->_in_channel_size);
		}
		_activations[id] = activation;
	}
	abcdl::algebra::Mat& get_activation(size_t id){
		CHECK(id < _activations.size());
		return _activations[id];
	}
protected:
    size_t _rows;
    size_t _cols;
    size_t _in_channel_size;
    size_t _out_channel_size;
    real _alpha = 0.1;

    std::vector<abcdl::algebra::RandomMatrix<real>> _batch_weights;
    abcdl::algebra::RandomMatrix<real> _batch_bias;

    std::vector<abcdl::algebra::RandomMatrix<real>> _delta_weights;
    abcdl::algebra::RandomMatrix<real> _delta_bias;

    std::vector<abcdl::algebra::RandomMatrix<real>> _weights;
    abcdl::algebra::RandomMatrix<real> _bias;

    abcdl::algebra::MatrixHelper<real> mh;

private:
    std::vector<abcdl::algebra::Mat&> _activations;
    std::vector<abcdl::algebra::Mat&> _deltas;
	abcdl::cnn::Layer_type;
};//class Layer

class InputLayer : public Layer{
public:
    InputLayer(const size_t rows, const size_t cols) : Layer(rows, cols, 1, 1, abcdl::cnn::INPUT){}
   
    void initialized(Layer* pre_layer){}	
	void forward(Layer* pre_layer){}
    void backward(Layer* pre_layer, Layer* back_layer){}

    void set_x(const abcdl::algebra::Mat& x){
        CHECK(x->get_rows() == this->_rows && x->get_cols() == this->_cols){
        this->set_activations(x, 0);
    }
};//class InputLayer

class SubSamplingLayer : public Layer{
public:
    SubSamplingLayer(const size_t scale, abcdl::cnn::Pooling* pooling) : Layer(0, 0, 0, 0, abcdl::cnn::SUBSAMPLING){
        _scale = scale;
        _pooling = pooling;
    }
    ~SubSamplingLayer(){
        delete _pooling;
    }

    inline size_t get_scale() const{ return _scale; }

    void initialized(Layer* pre_layer);	
    void forward(Layer* pre_layer);
    void backward(Layer* pre_layer, Layer* back_layer);

private:
    size_t _scale;
	abcdl::cnn::Pooling* _pooling;
};//class SubsamplingLayer


class ConvolutionLayer : public Layer{
public:
     ConvolutionLayer(const size_t kernal_size,
					  const size_t stride,
					  const size_t out_channel_size) : Layer(0, 0, 1, out_channel_size, abcdl::cnn::CONVOLUTION){
        _kernal_size = kernal_size;
        _stride = stride;
    }
	
    inline size_t get_stride() const { return _stride; }

    void initialized(Layer* pre_layer);	
    void forward(Layer* pre_layer);
    void backward(Layer* pre_layer, Layer* back_layer);

private:
    size_t _stride;
    size_t _kernal_size;
};//class ConvolutionLayer

class OutputLayer : public Layer{
public:
    OutputLayer(const size_t rows) : Layer(rows, 0, 0, 1, abcdl::cnn::OUTPUT){}
    
    void initialized(Layer* pre_layer);	
	void forward(Layer* pre_layer);
    void backward(Layer* pre_layer, Layer* back_layer);

    void set_y(const abcdl::algebra::Mat& y){
        CHECK(y.rows() == _rows);
        _y = y;
    }

private:
    abcdl::algebra::Mat _y;
    abcdl::algebra::Mat _pre_activation_array;
    real _loss;
};//class FullConnectionLayer

}//namespace cnn
}//namespace abcdl
