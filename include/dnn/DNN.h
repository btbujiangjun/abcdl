/***********************************************
 * Author: Jun Jiang - jiangjun4@sina.com
 * Create: 2017-08-15 17:40
 * Last modified : 2017-08-15 17:40
 * Filename      : DNN.h
 * Description   : 
 **********************************************/

#pragma once

namespace abcdl{
namespace dnn{

class DNN{
public:
    DNN();

    void set_layers(std::vector<Layer*>& layers){
        _layers = layers;
    }
    void set_epoch(const uint epoch){
        _epoch = epoch;
    }
    void set_alpha(const real alpha){
        _alpha = alpha;
    }
    void set_lamda(const real lamda){
        _lamda = lamda;
    }
    void set_batch_size(const uint batch_size){
        _batch_size = batch_size;
    }

    void train();
    void predict();

    bool load_model(const std::string& path);
    bool write_model(const std::string& path);

private:
    uint _epoch = 5;
    real _alpha = 0.5;
    real _lamda = 0.0;
    uint _batch_size = 1;
    std::vector<Layer*> _layers;
};//class DNN

}
}//namespace abcdl
