/***********************************************
 * Author: Jun Jiang - jiangjun4@sina.com
 * Create: 2017-08-15 17:40
 * Last modified : 2017-09-01 10:31
 * Filename      : FNN.h
 * Description   : Deep Neural Net 
 **********************************************/

#pragma once

#include <vector>
#include "algebra/Matrix.h"
#include "fnn/Layer.h"
#include "framework/Loss.h"
#include "utils/Log.h"
#include "utils/ModelLoader.h"

namespace abcdl{
namespace fnn{

class FNN{
public:
    FNN(){
        _loss = new abcdl::framework::MSELoss();
    }
    ~FNN(){
        for(auto& layer : _layers){
            delete layer;
			layer = nullptr;
        }
        _layers.clear();

        if(_loss != nullptr){
            delete _loss;
            _loss = nullptr;
        }
    }
    
	void set_epoch(const size_t epoch){ _epoch = epoch; }
    void set_alpha(const real alpha){ _alpha = alpha; }
    void set_lamda(const real lamda){ _lamda = lamda; }
    void set_batch_size(const size_t batch_size){ _batch_size = batch_size; }
    void set_loss_function(abcdl::framework::Loss* loss){
        if(_loss != nullptr){
            delete _loss;
        }
        _loss = loss;
    }

    void set_layers(std::vector<abcdl::fnn::Layer*>& layers){
        size_t layer_size = layers.size();
        CHECK(layer_size > 1 && layers[0]->get_layer_type() == abcdl::framework::INPUT);
        size_t output_dim = layers[0]->get_output_dim();
        for(size_t i = 1; i != layer_size; i++){
            CHECK(layers[i]->get_layer_type() == (i == layer_size - 1) ? abcdl::framework::OUTPUT : abcdl::framework::FULL_CONN);
            CHECK(output_dim == layers[i]->get_input_dim());
            output_dim = layers[i]->get_output_dim();
        }
        _layers = layers;
    }

    void train(const abcdl::algebra::Mat& train_data,
               const abcdl::algebra::Mat& train_label,
               const abcdl::algebra::Mat& test_data,
               const abcdl::algebra::Mat& test_label);
    void predict(abcdl::algebra::Mat& result, const abcdl::algebra::Mat& predict_data);

    bool load_model(const std::string& path);
    bool write_model(const std::string& path);

private:
    size_t evaluate(const abcdl::algebra::Mat& train_data,
                    const abcdl::algebra::Mat& train_label,
                    real* loss);

private:
    size_t _epoch = 30;
    real _alpha = 0.5;
    real _lamda = 0.1;
    size_t _batch_size = 50;
    std::vector<abcdl::fnn::Layer*> _layers;
    abcdl::framework::Loss* _loss;
    abcdl::utils::ModelLoader _model_loader;
};//class FNN

}//namespace fnn
}//namespace abcdl
