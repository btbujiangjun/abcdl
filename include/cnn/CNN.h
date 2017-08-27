/*********************************************
* Author: Jun Jiang - jiangjun4@sina.com
* Created: 2017-08-24 17:02
* Last modified: 2017-08-24 19:21
* Filename: CNN.h
* Description: Convolution Neural Network 
**********************************************/
#pragma once

#include <vector>
#include "algebra/Matrix.h"
#include "cnn/Layer.h"
#include "utils/Log.h"
#include "utils/ModelLoader.h"

namespace abcdl{
namespace cnn{
class CNN{
public:
    CNN(){}
    ~CNN(){
        for(auto layer : _layers){
            delete layer;
        }
        _layers.clear();
    }

	void set_epoch(const size_t epoch){ _epoch = epoch;}

    void set_layers(std::vector<abcdl::cnn::Layer*>& layers);
    
	void train(const abcdl::algebra::Mat& train_data,
               const abcdl::algebra::Mat& train_label,
               const abcdl::algebra::Mat& test_data,
               const abcdl::algebra::Mat& test_label);
    void predict(abcdl::algebra::Mat& result, const abcdl::algebra::Mat& predict_data);

	bool load_model(const std::string& path);
	bool write_model(const std::string& path);
private:
    bool evaluate(abcdl::algebra::Mat& data, abcdl::algebra::Mat& label);

private:
	size_t _epoch = 5;
	std::vector<abcdl::cnn::Layer*> _layers;

	abcdl::utils::ModelLoader _model_loader;
};//class CNN

}//namespace cnn
}//namespace abcdl
