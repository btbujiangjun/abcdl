/***********************************************
 * Author: Jun Jiang - jiangjun4@sina.com
 * Create: 2017-08-24 11:10
 * Last modified : 2017-08-24 15:56
 * Filename      : ModelLoader.h
 * Description   : 
 **********************************************/
#pragma once

#include <fstream>
#include <vector>
#include "algebra/Matrix.h"
#include "utils/Log.h"

namespace abcdl{
namespace utils{

class ModelInfo{
public:
    char type;
    size_t rows;
    size_t cols;
};//class ModelInfo

class ModelLoader{
public:
    template<class T>
    bool write(std::vector<abcdl::algebra::Matrix<T>*> models,
               const std::string& path,
               const std::string& signature = "",
               bool is_append = false);
    template<class T>
    bool write(abcdl::algebra::Matrix<T>* model,
               const std::string& path,
               const std::string& signature = "",
               bool is_append = false);
    template<class T>
    bool read(const std::string& path,
              std::vector<abcdl::algebra::Matrix<T>*>* models,
              const std::string& signature = "");
private:
    template<class T>
    bool generate_header(std::vector<abcdl::algebra::Matrix<T>*> models,
                         std::vector<ModelInfo>* infos);
};//class ModelLoader

template<class T>
bool ModelLoader::write(std::vector<abcdl::algebra::Matrix<T>*> models,
                        const std::string& path,
                        const std::string& signature,
                        bool is_append){
    std::vector<abcdl::algebra::Matrix<T>*> old_models;
    if(is_append){
        read<T>(path, &old_models, signature);
    }
    
    size_t num_models = models.size() + old_models.size();
    std::vector<ModelInfo> infos;

    if(!generate_header(old_models, &infos) || !generate_header(models, &infos)){
        return false;
    }
     
    std::ofstream out_file(path, std::ios::binary);

    out_file.write(signature.c_str(), sizeof(char)*signature.size());
    out_file.write((char*)&num_models, sizeof(size_t));
    for(auto&& info : infos){
        out_file.write(&info.type, sizeof(char));
        out_file.write((char*)&info.rows, sizeof(size_t));
        out_file.write((char*)&info.cols, sizeof(size_t));
    }

    size_t num_old_models = old_models.size();
    for(size_t i = 0; i != num_old_models; i++){
        out_file.write((char*)old_models[i]->data(), sizeof(T) * infos[i].rows * infos[i].cols);
    }

    num_models = models.size();
    for(size_t i = 0; i != num_models; i++){
        ModelInfo info = infos[i+num_old_models];
        out_file.write((char*)models[i]->data(), sizeof(T) * info.rows * info.cols);
    }

    out_file.close();

    return true;
}
template<class T>
bool ModelLoader::write(abcdl::algebra::Matrix<T>* model,
                        const std::string& path,
                        const std::string& signature,
                        bool is_append){
    std::vector<abcdl::algebra::Matrix<T>*> models;
    models.push_back(model);
    return write(models, path, is_append, signature);
}

template<class T>
bool ModelLoader::read(const std::string& path,
                       std::vector<abcdl::algebra::Matrix<T>*>* models,
                       const std::string& signature){
    std::ifstream in_file(path, std::ios::binary);
    if(!in_file){
        LOG(FATAL) << "Can't open Filename:" + path;
        return false;
    }

    size_t num_models = 0;
    std::string read_signature(signature.size(), ' ');
    in_file.read(&read_signature[0], sizeof(char) * signature.size());
    if(read_signature != signature){
        LOG(FATAL) << "ModelLoader read model error:[signature error]";
        in_file.close();
        return false;
    }

    in_file.read((char*)(&num_models), sizeof(size_t));

    std::vector<ModelInfo> infos;
    for(size_t i = 0; i != num_models; i++){
        ModelInfo info;
        in_file.read(&info.type, sizeof(char));
        in_file.read((char*)(&info.rows), sizeof(size_t));
        in_file.read((char*)(&info.cols), sizeof(size_t));
        infos.push_back(info);
    }

    models->clear();
    for(size_t i = 0; i != num_models; i++){
        ModelInfo info = infos[i];
        size_t size      = info.rows * info.cols;
        
        T* data = new T[size];
        in_file.read((char*)data, sizeof(T) * size);

        auto mat = new abcdl::algebra::Matrix<T>();
        mat->set_shallow_data(data, info.rows, info.cols);
        models->push_back(mat);
    }

    in_file.close();
    return true;
}

template<class T>
bool ModelLoader::generate_header(std::vector<abcdl::algebra::Matrix<T>*> models,
                                  std::vector<ModelInfo>* infos){
    for(auto&& model : models){
        ModelInfo info;
        info.rows = model->rows();
        info.cols = model->cols();
        if(typeid(T) == typeid(int)){
            info.type = 'i';
        }else if(typeid(T) == typeid(float)){
            info.type = 'f';
        }else if(typeid(T) == typeid(double)){
            info.type = 'd';
        }else{
            LOG(FATAL) << "ModelLoader not support data type";
            return false;
        }
        infos->push_back(info);
    }
    return true;
}

}//namespace utils
}//namespace abcdl