/*********************************************
* Author: Jun Jiang - jiangjun4@sina.com
* Created: 2017-04-18 17:41
* Last modified: 2017-09-01 10:53
* Filename: LibsvmHelper.h
* Description: read libsvm file
**********************************************/
#pragma once

#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include "utils/Log.h"
#include "utils/StringHelper.h"
#include "algebra/Matrix.h"
#include "algebra/MatrixSet.h"

namespace abcdl{
namespace utils{

template<class T>
class LibsvmHelper{
public:

    struct libsvm_sample{
        T label;
        std::vector<std::pair<size_t, T>> data;
    };

    bool read_data(const size_t feature_dim,
                   const std::string& file,
                   abcdl::algebra::Matrix<T>* out_data_mat,
                   abcdl::algebra::Matrix<T>* out_label_mat);
    bool read_data(const size_t feature_dim,
                   const size_t label_dim,
                   const std::string& file,
                   abcdl::algebra::Matrix<T>* out_data_mat,
                   abcdl::algebra::Matrix<T>* out_label_mat);
    bool write_data(const std::string& file,
                    const abcdl::algebra::Matrix<T>& data_mat,
                    const abcdl::algebra::Matrix<T>& label_mat);
private:
    size_t delta_size = 10000;
    abcdl::utils::StringHelper string_helper;
    void read_data_append_mat(abcdl::algebra::Matrix<T>* out_data_mat,
                              abcdl::algebra::Matrix<T>* out_label_mat,
                              const std::vector<libsvm_sample>& samples,
                              const size_t feature_dim,
                              const size_t label_dim){
        if(samples.size() == 0){
            return;
        }

        abcdl::algebra::Matrix<T> data_mat(samples.size(), feature_dim);
        abcdl::algebra::Matrix<T> label_mat(samples.size(), label_dim);
        for(int i = 0; i < samples.size(); i++){
            auto& sample = samples[i];
            label_mat.set_data(1, i, (size_t)sample.label);
            for(auto& pair : sample.data){
                if (pair.first >= feature_dim){
                    continue;
                }
                data_mat.set_data(pair.second, i, pair.first);
            }
        }
        out_data_mat->insert_row(data_mat);
        out_label_mat->insert_row(label_mat);

        LOG(INFO) << "Loading:" << samples.size() << "/" << out_label_mat->rows() << "/" << out_data_mat->rows() ;
    }
};//class LibsvmHelper

template<class T>
bool LibsvmHelper<T>::read_data(const size_t feature_dim,
                                const std::string& file,
                                abcdl::algebra::Matrix<T>* out_data_mat,
                                abcdl::algebra::Matrix<T>* out_label_mat){
    return read_data(feature_dim, 1, file, out_data_mat, out_label_mat);
}
template<class T>
bool LibsvmHelper<T>::read_data(const size_t feature_dim,
                                const size_t label_dim,
                                const std::string& file,
                                abcdl::algebra::Matrix<T>* out_data_mat,
                                abcdl::algebra::Matrix<T>* out_label_mat){

    LOG(INFO) << "Start load Libsvm file:" << file;
    std::vector<libsvm_sample> samples;
    std::string line;
    std::ifstream in_file(file);
    out_data_mat->clear();
    out_label_mat->clear();

    while(std::getline(in_file, line)){
        libsvm_sample sample;
        std::istringstream iss(line);
        iss >> sample.label;
        std::string ss;
        while(iss >> ss){
            std::vector<std::string> kv = string_helper.split(ss, ":");
            if (kv.size() != 2){
                continue;
            }
            sample.data.push_back(std::make_pair<int, T>(string_helper.str2int(kv[0]), string_helper.str2real(kv[1])));
        }
        samples.push_back(sample);
        
        if(samples.size() >= delta_size){
            read_data_append_mat(out_data_mat, out_label_mat, samples, feature_dim, label_dim);
            samples.clear();
        }

    }
        
    read_data_append_mat(out_data_mat, out_label_mat, samples, feature_dim, label_dim);
    samples.clear();

    return true;
}

template<class T>
bool LibsvmHelper<T>::write_data(const std::string& file,
                                 const abcdl::algebra::Matrix<T>& data_mat,
                                 const abcdl::algebra::Matrix<T>& label_mat){
    if(data_mat.rows() != label_mat.rows()){
        return false;
    }

    std::ofstream out_file(file);

    int row = data_mat.rows();
    int col = data_mat.cols();
    for(int i = 0; i < row; i++){
        out_file << label_mat.get_data(i, 0);
        for(int j = 0; j < col; j++){
            T value = data_mat.get_data(i, j);
            if(value!= 0){
                out_file << " " << j << ":" << value; 
            }
        }
        out_file << "\n";
    }
    return true;
}

}//namespace utils
}//namespace abcdl
