/***********************************************
 * Author: Jun Jiang - jiangjun4@sina.com
 * Create: 2017-08-19 18:50
 * Last modified : 2017-08-19 18:50
 * Filename      : DNN.cpp
 * Description   : 
 **********************************************/
#include "dnn/DNN.h"
#include "dnn/Layer.h"
#include "utils/Log.h"
#include "utils/Shuffler.h"

namespace abcdl{
namespace dnn{

void DNN::train(const abcdl::algebra::Mat& train_data, const abcdl::algebra::Mat& train_label){
    LOG(INFO) << "dnn start training...";

    size_t layer_size = _layers.size();
    size_t num_train_data = train_data.rows();

    CHECK(num_train_data == train_label.rows());
    CHECK(train_data.cols() == _layers[0]->get_input_dim());
    CHECK(train_label.cols() == _layers[_layers.size() -1]->get_output_dim());

    abcdl::utils::Shuffler shuffler(num_train_data);
    abcdl::algebra::Mat data;
    abcdl::algebra::Mat label;

    for(size_t i = 0; i != _epoch; i++){
        shuffler.shuffle();
        for(size_t j = 0; j != num_train_data; j++){
            train_data.get_row(&data, shuffler.get_row(j));
            train_label.get_row(&label, shuffler.get_row(j));

            for(size_t k = 0; k != layer_size; k++){
                if(k == 0){
                    _layers[k]->forward(data);
                }else{
                    _layers[k]->forward(_layers[k - 1]->get_activate_data());
                }
            }

            for(size_t k = layer_size - 1; k > 0; k--){
                if(k == layer_size - 1){
                    ((OutputLayer*)_layers[k])->set_y(label);
                    _layers[k]->backward(_layers[k - 1], nullptr);
                }else{
                    _layers[k]->backward(_layers[k -1], _layers[k + 1]);
                }
               
                //mini_batch_update
                if(j % _batch_size == _batch_size - 1 || j == num_train_data - 1){
                    _layers[k]->update_gradient(j % _batch_size, _alpha);
                }
            }

            if(j % 100 == 0){
                printf("Epoch[%ld/%ld] Train[%ld/%ld]\r", i, _epoch, j, num_train_data);
            }
        }
    }
}

void DNN::predict(const abcdl::algebra::Mat& predict_data){
    
}

bool DNN::load_model(const std::string& path){
    return true;
}
bool DNN::write_model(const std::string& path){
    return true;
}

}//namespace dnn
}//namespace abcdl
