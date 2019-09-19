/***********************************************
 * Author: Jun Jiang - jiangjun4@sina.com
 * Create: 2017-08-19 18:50
 * Last modified : 2017-08-19 18:50
 * Filename      : FNN.cpp
 * Description   : 
 **********************************************/
#include "fnn/FNN.h"
#include "fnn/Layer.h"
#include "utils/Log.h"
#include "utils/Shuffler.h"
#include <vector>


namespace abcdl{
namespace fnn{

void FNN::train(const abcdl::algebra::Mat& train_data,
                const abcdl::algebra::Mat& train_label){
    LOG(INFO) << "fnn start training...";

    size_t layer_size = _layers.size();
    size_t num_train_data = train_data.rows();

    CHECK(num_train_data == train_label.rows());
    CHECK(train_data.cols() == _layers[0]->get_input_dim());
    CHECK(train_label.cols() == _layers[_layers.size() -1]->get_output_dim());

    abcdl::algebra::Mat data;
    abcdl::algebra::Mat label;
    abcdl::utils::Shuffler shuffler(num_train_data);
    auto now = []{return std::chrono::system_clock::now();};

    for(size_t i = 0; i != _epoch; i++){
        
        shuffler.shuffle();
        auto start_time = now();
        real total_loss = 0;

        for(size_t j = 0; j != num_train_data; j++){
            train_data.get_row(&data, shuffler.get(j));
            train_label.get_row(&label, shuffler.get(j));

            for(size_t k = 0; k != layer_size; k++){
                if(k == 0){
                    ((InputLayer*)_layers[k])->set_x(data);
                }
                _layers[k]->forward(_layers[k-1]);
            }

            for(size_t k = layer_size - 1; k > 0; k--){
                if(k == layer_size - 1){
                    ((OutputLayer*)_layers[k])->set_y(label);
                    _layers[k]->backward(_layers[k-1], nullptr);
                }else{
                    _layers[k]->backward(_layers[k-1], _layers[k+1]);
                }
               
                //mini_batch_update
                if(j % _batch_size == _batch_size - 1 || j == num_train_data - 1){
                    _layers[k]->update_gradient(j % _batch_size + 1, _alpha, _lamda);
                }
            }

            total_loss += _loss->loss(label, _layers[layer_size - 1]->get_activate_data());

            if(j % 100 == 0){
                printf("Epoch[%ld/%ld] Train[%ld/%ld]\r", i + 1, _epoch, j, num_train_data);
            }
        }

        auto train_time = now();
        printf("Epoch[%ld] loss[%f] training run time:[%lld]ms\n", i + 1, total_loss/num_train_data, std::chrono::duration_cast<std::chrono::milliseconds>(train_time - start_time).count());
    }
}

size_t FNN::evaluate(const abcdl::algebra::Mat& test_data,
                     const abcdl::algebra::Mat& test_label,
                     real* loss){
    CHECK(test_data.rows() > 0);
    CHECK(test_data.cols() == _layers[0]->get_input_dim());
    CHECK(test_data.rows() == test_label.rows());
    CHECK(test_label.cols() == _layers[_layers.size() - 1]->get_output_dim());

    auto now = []{return std::chrono::system_clock::now();};
    auto start_time = now();
    
    size_t rows = test_data.rows();
    size_t predict_num = 0;
    abcdl::algebra::Mat mat;

    real total_loss = 0;
    for(size_t i = 0; i != rows; i++){
        predict(mat, test_data.get_row(i));
        if (i >= 0){
            //printf("%f/%f\n", mat.get_data(0, 0) , test_label.get_row(i).get_data(0, 0));
            //mat.display("|");
            //test_label.get_row(i).display("^");
            //test_data.get_row(i).display("~");
            //printf("\n");
        }
        if(mat.cols() == 1 && mat == test_label.get_row(i)){
            ++ predict_num; 
        }else if(mat.argmax() == test_label.get_row(i).argmax()){
            ++ predict_num;
        }
        total_loss += _loss->loss(test_label.get_row(i), mat);
        mat.clear();
    }
    *loss = total_loss / rows;
    
    LOG(INFO) << "loss[" << *loss << "] ["<< predict_num <<"/" << rows << "] rate[" << predict_num/(real)rows << "]";
    auto predict_time = now();
    printf("evalate loss[%f] rate[%f] [%zu/%zu]predict run time:[%lld]ms\n", *loss, predict_num/(real)rows, predict_num, rows, std::chrono::duration_cast<std::chrono::milliseconds>(predict_time - start_time).count());
    return predict_num;
}

void FNN::predict(abcdl::algebra::Mat& result, const abcdl::algebra::Mat& predict_data){
    CHECK(predict_data.cols() == _layers[0]->get_input_dim());
    size_t layer_size = _layers.size();
    for(size_t k = 0; k < layer_size; k++){
        if(k == 0){
            ((InputLayer*)_layers[k])->set_x(predict_data);
        }
        _layers[k]->forward(_layers[k-1]);
        if(k > 0){
            //_layers[k]->get_activate_data().display("|");
        }
    }
    //_layers[layer_size - 1]->get_activate_data().display("|");
    result = _layers[layer_size - 1]->get_activate_data();
}

bool FNN::load_model(const std::string& path){
    _path = path;
    LOG(INFO) << "loading model from:" << path;
    std::vector<abcdl::algebra::Mat*> params;
    if(!_model_loader.read<real>(path, &params, "FNNMODEL") || params.size() % 2 != 0){
        if(params.size() == 0){
            LOG(INFO) << "Train with new model";
            return false;
        }
        for(auto& param : params){
            delete param;
        }
        params.clear();
        LOG(FATAL) << "Load model error:" << params.size() << " must be:" << _layers.size();
        return false;
    }
    
    size_t size = params.size() / 2;
    if(_layers.size() - 1 != size){
        LOG(FATAL) << "Model size is error:" << size << " must be:" << _layers.size() - 1;
        return false;
    }

    for(int i = 1; i < _layers.size(); i++){
        if(!_layers[i]->set_weight(*params[2 * (i-1)])){
            LOG(FATAL) << "Set Layer weight error, layer: " << i;
            return false;
        }
        if(!_layers[i]->set_bias(*params[2 * (i - 1) + 1])){
            LOG(FATAL) << "Set Layer bias error, layer: " << i;
            return false;
        }
    }

    for(auto& param : params){
        delete param;
    }
    params.clear();
    LOG(INFO) << "Load model success:" << path;
    
    return true;
}

bool FNN::write_model(const std::string& path){
    std::vector<abcdl::algebra::Mat*> params;
    for(size_t i = 1; i != _layers.size(); i++){
        params.push_back(&_layers[i]->get_weight());
        params.push_back(&_layers[i]->get_bias());
    }
    return _model_loader.write<real>(params, path, "FNNMODEL", false);
}

}//namespace fnn
}//namespace abcdl
