/*********************************************
* Author: Jun Jiang - jiangjun4@sina.com
* Created: 2017-08-24 17:02
* Last modified: 2017-08-31 11:02
* Filename: CNN.h
* Description: Convolution Neural Network 
**********************************************/
#pragma once

#include <vector>
#include "algebra/Matrix.h"
#include "algebra/MatrixSet.h"
#include "cnn/Layer.h"
#include "utils/Log.h"
#include "utils/ModelLoader.h"

namespace abcdl{
namespace cnn{

class CNN{
public:
    CNN(){}
    ~CNN(){
        for(auto layer : _layers){delete layer;}
        _layers.clear();
    }

	void set_epoch(const size_t epoch){_epoch = epoch;}
    void set_alpha(const real alpha){_alpha = alpha;}
    void set_batch_size(const size_t batch_size){_batch_size = batch_size;}
    void set_layers(std::vector<abcdl::cnn::Layer*> layers);
    
    void train(const abcdl::algebra::MatSet& train_data,
               const abcdl::algebra::MatSet& train_label,
               const abcdl::algebra::MatSet& test_data,
               const abcdl::algebra::MatSet& test_label);
    void predict(abcdl::algebra::Mat& result, const abcdl::algebra::Mat& predict_data);

	bool load_model(const std::string& path){return true;}
	bool write_model(const std::string& path){return true;}
private:
    size_t evaluate(const abcdl::algebra::MatSet& data, const abcdl::algebra::MatSet& label);
    void forward(const abcdl::algebra::Mat& mat);
    void backward(const abcdl::algebra::Mat& mat);
    void update_gradient(const size_t batch_size, const real alpha);

    bool check(const size_t rows, const size_t cols) const;
private:
	size_t _epoch = 50;
    size_t _batch_size = 1;
    real _alpha = 0.1f;
	std::vector<abcdl::cnn::Layer*> _layers;

	//abcdl::utils::ModelLoader _model_loader;
};//class CNN

}//namespace cnn
}//namespace abcdl
