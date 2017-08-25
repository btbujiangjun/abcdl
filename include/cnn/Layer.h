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

class Layer{
public:
    Layer(size_t rows,
          size_t cols,
          size_t in_channel_size,
          size_t out_channel_size) : _rows(rows),
        _cols(cols),
        _in_channel_size(in_channel_size),
        _out_channel_size(out_channel_size),
    	_is_last_layer(true){}

    virtual ~Layer(){
        if(_bias != nullptr){
            delete _bias;
            _bias = nullptr;
        }

        clear_vector_matrix(&_weights);
        clear_vector_matrix(&_activations);
        clear_vector_matrix(&_deltas);
    }

    virtual bool initialize(Layer* pre_layer = nullptr) = 0;
    virtual void feed_forward(Layer* pre_layer = nullptr, bool debug = false) = 0;
    virtual void back_propagation(Layer* pre_layer, Layer* back_layer = nullptr, bool debug = false) = 0;

    inline void set_rows(size_t rows){_rows = rows;}
    inline size_t get_rows(){return _rows;}

    inline void set_cols(size_t cols){_cols = cols;}
    inline size_t get_cols(){return _cols;}

    inline void set_is_last_layer(){_is_last_layer = false;}
    inline bool get_is_last_layer(){return _is_last_layer;}
    /*
     * pre_layer feature channel size
     */
    inline void set_in_channel_size(size_t in_channel_size){_in_channel_size = in_channel_size;}
    inline size_t get_in_channel_size(){return _in_channel_size;}

    /*
     * current_layer feature channel size
     */
    inline void set_out_channel_size(size_t out_channel_size){_out_channel_size = out_channel_size;}
    inline size_t get_out_channel_size(){return _out_channel_size;}

    //activation size equal out_channel_size
    inline void set_activation(size_t out_channel_id, abcdl::algebra::Matccma::algebra::BaseMatrixT<real>* activation){
        set_vec_mat(&_activations, out_channel_id, activation);
    }
    inline abcdl::algebra::Matccma::algebra::BaseMatrixT<real>* get_activation(size_t out_channel_id){
        return _activations[out_channel_id];
    }

    //delta size equal out_channel_size
    inline void set_delta(size_t out_channel_id, abcdl::algebra::Matccma::algebra::BaseMatrixT<real>* delta){
        set_vec_mat(&_deltas, out_channel_id, delta);
    }
    inline abcdl::algebra::Matccma::algebra::BaseMatrixT<real>* get_delta(size_t out_channel_id){
        return _deltas[out_channel_id];
    }

    //weight size equal in_channel_size * out_channel_size
    inline void set_weight(size_t in_channel_id,
                           size_t out_channel_id,
                           abcdl::algebra::Matccma::algebra::BaseMatrixT<real>* weight){
        set_vec_mat(&_weights, in_channel_id * this->_out_channel_size + out_channel_id, weight);
    }
    inline abcdl::algebra::Matccma::algebra::BaseMatrixT<real>* get_weight(size_t in_channel_id, size_t out_channel_id){
        return _weights[in_channel_id * this->_out_channel_size + out_channel_id];
    }

    inline void set_bias(abcdl::algebra::Matccma::algebra::BaseMatrixT<real>* bias){
        if(_bias != nullptr){
            delete _bias;
        }
        _bias = bias;
    }
    inline abcdl::algebra::Matccma::algebra::BaseMatrixT<real>* get_bias(){ return _bias;}

private:
    inline void clear_vector_matrix(std::vector<abcdl::algebra::Matccma::algebra::BaseMatrixT<real>*>* vec_mat){
        for(auto mat : *vec_mat){
            delete mat;
            mat = nullptr;
        }
        vec_mat->clear();
    }

    inline void set_vec_mat(std::vector<abcdl::algebra::Matccma::algebra::BaseMatrixT<real>*>* vec_mat, size_t idx, abcdl::algebra::Matccma::algebra::BaseMatrixT<real>* mat){
        size_t size = vec_mat->size();
        if(size > idx){
            clear_vector_matrix(vec_mat);
        }
        if(vec_mat->size() == idx){
            vec_mat->push_back(mat);
        }else{
            printf("set_vec_mat error:[%d/%d]\n", size, idx);
        }
    }

protected:
    size_t _rows;
    size_t _cols;
    /*pre_layer feature channel size*/
    size_t _in_channel_size;
    /* current_layer feature channel size*/
    size_t _out_channel_size;
    std::vector<abcdl::algebra::Mat*> _weights;
    abcdl::algebra::Mat _bias;

    real _alpha = 0.1;
private:
    std::vector<abcdl::algebra::Mat*> _activations;
    std::vector<abcdl::algebra::Mat*> _deltas;

    bool _is_last_layer = true;
};//class Layer

class DataLayer:public Layer{
public:
    DataLayer(size_t rows, size_t cols):Layer(rows, cols, 1, 1){}
    ~DataLayer(){
        _x = nullptr; //out pointer,no delete
    }
    bool initialize(Layer* pre_layer = nullptr);
    void feed_forward(Layer* pre_layer = nullptr, bool debug = false);
    void back_propagation(Layer* pre_layer, Layer* back_layer = nullptr, bool debug = false);

    bool set_x(abcdl::algebra::Matccma::algebra::BaseMatrixT<real>* x){
        if(x->get_rows() == this->_rows && x->get_cols() == this->_cols){
            _x = x;
            return true;
        }
        printf("DataLayer mat dim error\n");
        return false;
    }
private:
    abcdl::algebra::Matccma::algebra::BaseMatrixT<real>* _x;
};//class DataLayer

class SubSamplingLayer:public Layer{
public:
    SubSamplingLayer(size_t scale, Pooling* pooling):Layer(0, 0, 0, 0){
        _scale = scale;
        _pooling = pooling;
    }
    ~SubSamplingLayer(){
        delete _pooling;
    }
    bool initialize(Layer* pre_layer = nullptr);
    void feed_forward(Layer* pre_layer = nullptr, bool debug = false);
    void back_propagation(Layer* pre_layer, Layer* back_layer = nullptr, bool debug = false);

    size_t get_scale(){return _scale;}

protected:
    size_t _scale;
private:
    Pooling* _pooling;
};//class SubsamplingLayer


class ConvolutionLayer : public Layer{
public:
     ConvolutionLayer(size_t kernal_size, size_t stride, size_t out_channel_size):Layer(0, 0, 1, out_channel_size){
        _kernal_size = kernal_size;
        _stride = stride;
    }
    inline size_t get_stride()const {return _stride;}

    bool initialize(Layer* pre_layer = nullptr);
    void feed_forward(Layer* pre_layer = nullptr, bool debug = false);
    void back_propagation(Layer* pre_layer, Layer* back_layer = nullptr, bool debug = false);
protected:
    size_t _stride;
    size_t _kernal_size;
};//class ConvolutionLayer

class FullConnectionLayer : public Layer{
public:
    FullConnectionLayer(size_t rows):Layer(rows, 0, 0, 1){}
    ~FullConnectionLayer(){
        if(_av != nullptr){
            delete _av;
            _av = nullptr;
        }
        if(_error != nullptr){
            delete _error;
            _error = nullptr;
        }

        _y = nullptr;//out pointer, not delete data.
    }
    bool initialize(Layer* pre_layer = nullptr);
    void feed_forward(Layer* pre_layer = nullptr, bool debug = false);
    void back_propagation(Layer* pre_layer, Layer* back_layer = nullptr, bool debug = false);

    bool set_y(abcdl::algebra::Matccma::algebra::BaseMatrixT<real>* y){
        if(y->get_rows() == _rows){
            _y = y;
            return true;
        }
        return false;
    }

    real get_loss() const{
        return _loss;
    }
private:
    abcdl::algebra::Mat _y;
    abcdl::algebra::Mat _av;//pre_layer activations' vector
    abcdl::algebra::Mat _error;
    real _loss;
};//class FullConnectionLayer

}//namespace cnn
}//namespace ccma
