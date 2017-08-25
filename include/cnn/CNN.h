/*********************************************
* Author: Jun Jiang - jiangjun4@sina.com
* Created: 2017-08-24 17:02
* Last modified: 2017-08-24 19:21
* Filename: CNN.h
* Description: Convolution Neural Network 
**********************************************/
#pragma once

#include "algebra/Matrix.h"
#include "include/cnn/Layer.h"

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

    bool add_layer(Layer* layer);
    void train(const abcdl::algebra::Mat& train_data,
               const abcdl::algebra::Mat& train_label,
               const size_t epoch = 1,
               const abcdl::algebra::Mat& test_data,
               const abcdl::algebra::Mat& test_label);
    //void predict(const abcdl::algebra::Mat& predict_data);
protected:
    void feed_forward(abcdl::algebra::Mat& mat);
    void back_propagation(abcdl::algebra::Mat& mat);

private:
    bool check(uint size);
    bool evaluate(abcdl::algebra::Mat& data, abcdl::algebra::Mat& label);

private:
    std::vector<Layer*> _layers;
};//class CNN

}//namespace cnn
}//namespace abcdl
