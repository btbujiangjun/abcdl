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
#include <algorithm>

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
    auto start_time = now();
    real total_loss = 0;
    std::vector<std::pair<real, real>> auc_train_vec;

    shuffler.shuffle();
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
                _layers[k]->update_gradient(j % _batch_size + 1, _alpha);
            }
        }

        total_loss += _loss->loss(label, _layers[layer_size - 1]->get_activate_data());
        if(_layers[layer_size - 1]->get_activate_data().cols() == 2){
            auc_train_vec.push_back(std::make_pair(label.get_data(0, 1), _layers[layer_size - 1]->get_activate_data().get_data(0, 1)));
        }

        if(j % 100 == 0){
            printf("Train[%ld/%ld]\r", j, num_train_data);
        }
    }
        
    double auc_score = auc(auc_train_vec);
    double avg_loss = total_loss / num_train_data;
    long long int train_time = std::chrono::duration_cast<std::chrono::milliseconds>(now() - start_time).count();

    LOG(INFO) << "auc["<< auc_score <<"] loss["<< avg_loss <<"] training run time:["<< train_time <<"]ms";
    printf("auc[%lf] loss[%lf] training run time:[%lld]ms\n", auc_score, avg_loss, train_time);
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
    std::vector<std::pair<real, real>> auc_evaluate_vec;

    for(size_t i = 0; i < rows; i++){
        predict(mat, test_data.get_row(i));
        if(mat.cols() == 1 && mat == test_label.get_row(i)){
            ++ predict_num; 
        }else if(mat.argmax() == test_label.get_row(i).argmax()){
            ++ predict_num;
        }
        if(mat.cols() == 2){
            auc_evaluate_vec.push_back(std::make_pair(test_label.get_row(i).get_data(0, 1), mat.get_data(0, 1)));
        }
        total_loss += _loss->loss(test_label.get_row(i), mat);
    }

    *loss = total_loss / rows;
    double auc_score = auc(auc_evaluate_vec); 
    LOG(INFO) << "evaluate auc[" << auc_score << "] loss[" << *loss << "] ["<< predict_num <<"/" << rows << "] rate[" << predict_num/(real)rows << "]";
    auto predict_time = now();
    printf("evaluate auc[%lf] loss[%f] rate[%f] [%zu/%zu]predict run time:[%lld]ms\n", auc_score, *loss, predict_num/(real)rows, predict_num, rows, (long long int)std::chrono::duration_cast<std::chrono::milliseconds>(predict_time - start_time).count());
    return predict_num;
}

double FNN::auc(std::vector<std::pair<real,real>>& auc_vec){
    if(auc_vec.size() == 0){
        return 0;
    }
    std::sort(auc_vec.begin(), auc_vec.end(), [](const std::pair<real, real> a, const std::pair<real, real> b){return a.second < b.second;});
    size_t m = 0;
    double rank = 0;
    for(size_t i = 0; i < auc_vec.size(); i++){
        //printf("i:%zu, label:%lf,value:%lf\n", i, auc_vec[i].first, auc_vec[i].second);
        if(auc_vec[i].first > 0){
            m += 1;
            rank += (i+1);
            //printf("m:%zu, rank:%lf\n", m, rank);
        }
    }
    //printf("M:%zu, N:%zu, rank:%lf\n", m, auc_vec.size() - m , rank);
    return (rank - m * (m + 1) / 2) / (m * (auc_vec.size() - m));
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
    //predict_data.display("^");
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

    for(size_t i = 1; i < _layers.size(); i++){
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
