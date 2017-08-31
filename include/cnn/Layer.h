/*********************************************
* Author: Jun Jiang - jiangjun4@sina.com
* Created: 2017-08-24 15:03
* Last modified: 2017-08-31 11:01
* Filename: Layer.h
* Description: CNN network layer
**********************************************/
#pragma once

#include <vector>
#include "framework/Pool.h"
#include "framework/Cost.h"
#include "framework/ActivateFunc.h"
#include "algebra/Matrix.h"
#include "algebra/MatrixHelper.h"
#include "utils/Log.h"

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

    virtual ~Layer(){
        for(auto& weight : _batch_weights){delete weight;}
        _batch_weights.clear();
        delete _batch_bias;

        for(auto& weight : _delta_weights){delete weight;}
        _delta_weights.clear();
        delete _delta_bias;

        for(auto& weight : _weights){delete weight;}
        _weights.clear();
        delete _bias;

        for(auto& delta : _deltas){delete delta;}
        _deltas.clear();

        for(auto& activation : _activations){delete activation;}
        _activations.clear();
    }

	size_t get_rows() const { return _rows; }
	size_t get_cols() const { return _cols; }

	inline size_t get_out_channel_size() const { return _out_channel_size; }
    inline abcdl::cnn::Layer_type get_layer_type() const {return _layer_type;}

	virtual void initialize(Layer* pre_layer) = 0;
    virtual void forward(Layer* pre_layer){ clear(); };
    virtual void backward(Layer* pre_layer, Layer* back_layer) = 0;

    void update_gradient(const size_t batch_size, const real alpha){
        real learning_rate = alpha / batch_size;
        for(size_t i = 0; i != _weights.size(); i++){
            _weights[i]->operator-=(_batch_weights[i]->operator*(learning_rate));
        }
        if(_bias->get_size() > 0){
            _bias->operator-=(_batch_bias->operator*(learning_rate));
        }

        for(auto& weight : _batch_weights){
            weight->reset();
        }
        _batch_bias->reset();
    }

	abcdl::algebra::Mat& get_activation(size_t id) const{
		CHECK(id < _activations.size());
		return *_activations[id];
	}
	inline abcdl::algebra::RandomMatrix<real>& get_weight(const size_t in_channel_id, const size_t out_channel_id) const{
        size_t size = in_channel_id * _out_channel_size + out_channel_id;
        CHECK(size < _weights.size());
		return *_weights[size];
	}
    abcdl::algebra::Mat& get_delta(const size_t id) const {
        CHECK(id < _deltas.size());
        return *_deltas[id];
    }

protected:
	inline void set_delta_weight(const size_t in_channel_id,
						         const size_t out_channel_id,
                                 const abcdl::algebra::Mat& weight){
		CHECK(in_channel_id < _in_channel_size && out_channel_id < _out_channel_size);
        (*_delta_weights[in_channel_id * _out_channel_size + out_channel_id]) = weight;
	}

	abcdl::algebra::Mat& get_delta_weight(const size_t in_channel_id, const size_t out_channel_id){ 
		CHECK(in_channel_id < _in_channel_size && out_channel_id < _out_channel_size);
        size_t size = in_channel_id * _out_channel_size + out_channel_id;
        return *_delta_weights[size];
	}

	inline void set_batch_weight(const size_t in_channel_id,
						         const size_t out_channel_id,
       						     const abcdl::algebra::Mat& weight){
		CHECK(in_channel_id < _in_channel_size && out_channel_id < _out_channel_size);
        (*_batch_weights[in_channel_id * _out_channel_size + out_channel_id]) += weight;
	}

    void set_delta(const size_t id, const abcdl::algebra::Mat& delta){
        CHECK(id < _out_channel_size);
        (*_deltas[id]) = delta;
    }

	void set_activation(const size_t id, const abcdl::algebra::Mat& activation){
		CHECK(id < _out_channel_size);
        (*_activations[id]) = activation;
	}

protected:
    size_t _rows;
    size_t _cols;
    size_t _in_channel_size;
    size_t _out_channel_size;
    
    std::vector<abcdl::algebra::RandomMatrix<real>*> _weights;
    abcdl::algebra::Mat* _bias = new abcdl::algebra::Mat();
    std::vector<abcdl::algebra::Mat*> _activations;

    std::vector<abcdl::algebra::Mat*> _delta_weights;
    abcdl::algebra::Mat* _delta_bias = new abcdl::algebra::Mat();

    std::vector<abcdl::algebra::Mat*> _batch_weights;
    abcdl::algebra::Mat* _batch_bias = new abcdl::algebra::Mat();

    std::vector<abcdl::algebra::Mat*> _deltas;

    abcdl::algebra::MatrixHelper<real> mh;
private:
    inline void clear(){
        for(auto& weight : _delta_weights){
            weight->reset();
        }

        for(auto& delta : _deltas){
            delta->reset();
        }

        for(auto& activation : _activations){
            activation->reset();
        }
    }
private:
	abcdl::cnn::Layer_type _layer_type;
};//class Layer

class InputLayer : public Layer{
public:
    InputLayer(const size_t rows, const size_t cols) : Layer(rows, cols, 1, 1, abcdl::cnn::INPUT){}
   
    void initialize(Layer* pre_layer){_activations.push_back(new abcdl::algebra::Mat());}	
	void forward(Layer* pre_layer){}
    void backward(Layer* pre_layer, Layer* back_layer){}

    void set_x(const abcdl::algebra::Mat& x){
        CHECK(x.rows() == this->_rows && x.cols() == this->_cols);
        this->set_activation(0, x);
    }
};//class InputLayer

class SubSamplingLayer : public Layer{
public:
    SubSamplingLayer(const size_t scale, abcdl::framework::Pooling* pooling) : Layer(0, 0, 0, 0, abcdl::cnn::SUBSAMPLING){
        _scale = scale;
        _pooling = pooling;
    }
    ~SubSamplingLayer(){
        delete _pooling;
    }

    inline size_t get_scale() const{ return _scale; }

    void initialize(Layer* pre_layer);	
    void forward(Layer* pre_layer);
    void backward(Layer* pre_layer, Layer* back_layer);

private:
    size_t _scale;
	abcdl::framework::Pooling* _pooling;
};//class SubsamplingLayer


class ConvolutionLayer : public Layer{
public:
     ConvolutionLayer(const size_t kernal_size,
					  const size_t stride,
					  const size_t out_channel_size,
                      abcdl::framework::ActivateFunc* activate_func) : Layer(0, 0, 1, out_channel_size, abcdl::cnn::CONVOLUTION){
        _kernal_size = kernal_size;
        _stride = stride;
        _activate_func = activate_func;
    }

    ~ConvolutionLayer(){
        delete _activate_func;
    }
	
    inline size_t get_stride() const { return _stride; }

    void initialize(Layer* pre_layer);	
    void forward(Layer* pre_layer);
    void backward(Layer* pre_layer, Layer* back_layer);

private:
    size_t _stride;
    size_t _kernal_size;
    abcdl::framework::ActivateFunc* _activate_func;
};//class ConvolutionLayer

class OutputLayer : public Layer{
public:
    OutputLayer(const size_t rows,
                abcdl::framework::ActivateFunc* activate_func,
                abcdl::framework::Cost* cost) : Layer(rows, 0, 0, 1, abcdl::cnn::OUTPUT){
        _activate_func = activate_func;
        _cost = cost;
    }
    ~OutputLayer(){
        delete _cost;
        delete _activate_func;
    }

    void initialize(Layer* pre_layer);	
	void forward(Layer* pre_layer);
    void backward(Layer* pre_layer, Layer* back_layer);

    void set_y(const abcdl::algebra::Mat& y){
        CHECK(y.rows() == _rows);
        _y = y;
    }
    real get_loss() const{
        return _loss;
    }

private:
    real _loss;
    abcdl::algebra::Mat _y;
    abcdl::algebra::Mat _pre_activation_array;
    abcdl::framework::Cost* _cost;
    abcdl::framework::ActivateFunc* _activate_func;
};//class FullConnectionLayer

}//namespace cnn
}//namespace abcdl
