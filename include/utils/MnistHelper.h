/*********************************************
* Author: Jun Jiang - jiangjun4@sina.com
* Created: 2017-04-18 17:41
* Last modified: 2017-09-01 10:53
* Filename: MnistHelper.h
* Description: read mnist file
**********************************************/
#pragma once

#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include "algebra/Matrix.h"
#include "algebra/MatrixSet.h"

namespace abcdl{
namespace utils{

template<class T>
class MnistHelper{
public:
    bool read_image(const std::string& image_file,
                    abcdl::algebra::Matrix<T>* out_mat,
                    const int limit = -1,
                    const size_t threshold = 0);

    bool read_images(const std::string& image_file,
                     abcdl::algebra::MatrixSet<T>& out_matrix_set,
                     const int limit = -1,
                     const size_t threshold = 0);

    bool read_label(const std::string& label_file,
                    abcdl::algebra::Matrix<T>* out_mat,
                    const int limit = -1);

    bool read_vec_label(const std::string& label_file,
                        abcdl::algebra::Matrix<T>* out_mat,
                        const int limit = -1,
                        const size_t vec_size = 10);

    bool read_vec_labels(const std::string& label_file,
                         abcdl::algebra::MatrixSet<T>& out_matrix_set,
                         const int limit = -1,
                         const size_t vec_size = 10);

private:
    inline std::unique_ptr<char[]> read_mnist_file(const std::string& path,
                                                   const int key,
                                                   size_t* out_rows);
    inline int read_header(const std::unique_ptr<char[]>& buffer, size_t position);
    inline T* vectorize_label(const size_t data, const size_t sizes);

};//class MnistHelper

template<class T>
class MnistReader{
public:
    MnistReader(const std::string& dir){
        _dir = dir;
    }
    MnistReader(const std::string& dir, const std::string& dataset){
        _dir = dir;
        _dataset = dataset;
    }
    virtual bool read_train_image(abcdl::algebra::Matrix<T>* out_mat,
                                  const int limit = -1,
                                  const size_t threshold = 0){
        return _helper.read_image(_dir + "/" + _dataset + "/train-images-idx3-ubyte", out_mat, limit, threshold); 
    }
    virtual bool read_train_images(abcdl::algebra::MatrixSet<T>& out_matrix_set,
                                   const int limit = -1,
                                   const size_t threshold = 0){
        return _helper.read_images(_dir + "/" + _dataset + "/train-images-idx3-ubyte", out_matrix_set, limit, threshold); 
    }
    virtual bool read_test_image(abcdl::algebra::Matrix<T>* out_mat,
                                 const int limit = -1,
                                 const size_t threshold = 0){
        return _helper.read_image(_dir + "/" + _dataset + "/t10k-images-idx3-ubyte", out_mat, limit, threshold); 
    }
    virtual bool read_test_images(abcdl::algebra::MatrixSet<T>& out_matrix_set,
                                  const int limit = -1,
                                  const size_t threshold = 0){
        return _helper.read_images(_dir + "/" + _dataset + "/t10k-images-idx3-ubyte", out_matrix_set, limit, threshold); 
    }
    virtual bool read_train_label(abcdl::algebra::Matrix<T>* out_mat, const int limit = -1){
        return _helper.read_label(_dir + "/" + _dataset + "/train-labels-idx1-ubyte", out_mat, limit);
    }
    virtual bool read_test_label(abcdl::algebra::Matrix<T>* out_mat, const int limit = -1){
        return _helper.read_label(_dir + "/" + _dataset + "/t10k-labels-idx1-ubyte", out_mat, limit);
    }
    virtual bool read_train_vec_label(abcdl::algebra::Matrix<T>* out_mat,
                                      const int limit = -1,
                                      const size_t vec_size = 10){
        return _helper.read_vec_label(_dir + "/" + _dataset + "/train-labels-idx1-ubyte", out_mat, limit);
    }
    virtual bool read_train_vec_labels(abcdl::algebra::MatrixSet<T>& out_matrix_set,
                                       const int limit = -1,
                                       const size_t vec_size = 10){
        return _helper.read_vec_labels(_dir + "/" + _dataset + "/train-labels-idx1-ubyte", out_matrix_set, limit);
    }
    virtual bool read_test_vec_label(abcdl::algebra::Matrix<T>* out_mat,
                                     const int limit = -1,
                                     const size_t vec_size = 10){
        return _helper.read_vec_label(_dir + "/" + _dataset + "/t10k-labels-idx1-ubyte", out_mat, limit);
    }
    virtual bool read_test_vec_labels(abcdl::algebra::MatrixSet<T>& out_matrix_set,
                                      const int limit = -1,
                                      const size_t vec_size = 10){
        return _helper.read_vec_labels(_dir + "/" + _dataset + "/t10k-labels-idx1-ubyte", out_matrix_set, limit);
    }

private:
    std::string _dir;
    std::string _dataset = "mnist";
    MnistHelper<T> _helper;
};//MnistReader

template<class T>
class FashionMnistReader : public MnistReader<T>{
public:
    FashionMnistReader(const std::string& dir) : MnistReader<T>(dir, "fashion-mnist"){}
};//class FashionMnistReader

template<class T>
bool MnistHelper<T>::read_image(const std::string& image_file,
                                abcdl::algebra::Matrix<T>* out_mat,
                                const int limit,
                                const size_t threshold){

    size_t count = 0;
    auto image_buffer = read_mnist_file(image_file, 0x803, &count);

    if(!image_buffer || count == 0){
        return false;
    }

    if( limit > 0 && limit < (int)count){
        count = limit;
    }

    //read image data
    auto rows = read_header(image_buffer, 2);
    auto cols = read_header(image_buffer, 3);

    auto image_data_buffer = reinterpret_cast<unsigned char*>(image_buffer.get() + 16);

    size_t size = count * rows * cols;
    T* data = new T[size];
    for(size_t i = 0; i < size; i++){
        if(threshold > 0){
            data[i] = static_cast<T>(*image_data_buffer++) > threshold ? 1 : 0;
        }else{
            data[i] = static_cast<T>(*image_data_buffer++) / 255;
        }
    }

    out_mat->set_shallow_data(data, count, rows * cols);

    return true;
}

template<class T>
bool MnistHelper<T>::read_images(const std::string& image_file,
                                 abcdl::algebra::MatrixSet<T>& out_matrix_set,
                                 const int limit,
                                 const size_t threshold){

    size_t count = 0;
    auto image_buffer = read_mnist_file(image_file, 0x803, &count);

    if(!image_buffer || count == 0){
        return false;
    }

    if( limit > 0 && limit < (int)count){
        count = limit;
    }

    //read image data
    auto rows = read_header(image_buffer, 2);
    auto cols = read_header(image_buffer, 3);

    auto image_data_buffer = reinterpret_cast<unsigned char*>(image_buffer.get() + 16);

    auto size = rows * cols;
    abcdl::algebra::Matrix<T> mat(rows, cols);
    for(size_t i = 0; i != count; i++){
        mat.reset();
        T* data = mat.data();
        for(int j = 0; j != size; j++){
            if(threshold > 0){
                data[j] = static_cast<T>(*image_data_buffer++) > threshold ? 1 : 0;
            }else{
                data[j] = static_cast<T>(*image_data_buffer++) / 255;
            }
        }
        out_matrix_set.push_back(mat);
    }

    return true;
}

template<class T>
bool MnistHelper<T>::read_label(const std::string& label_file,
                                abcdl::algebra::Matrix<T>* out_mat,
                                const int limit){
    size_t count = 0;
    auto label_buffer = read_mnist_file(label_file, 0x801, &count);
    if(!label_buffer || count == 0){
        return false;
    }

    if( limit > 0 && limit < (int)count){
        count = limit;
    }

    //read label data
    auto label_data_buffer = reinterpret_cast<unsigned char*>(label_buffer.get() + 8);

    T* labels = new T[count];
    for(size_t i = 0; i < count; i++){
        labels[i] = static_cast<T>(*label_data_buffer++);
    }

    out_mat->set_shallow_data(labels, count, 1);

    return true;
}

template<class T>
bool MnistHelper<T>::read_vec_label(const std::string& label_file,
                                    abcdl::algebra::Matrix<T>* out_mat,
                                    const int limit,
                                    const size_t vec_size){
    size_t count = 0;
    auto label_buffer = read_mnist_file(label_file, 0x801, &count);
    if(!label_buffer || count == 0){
        return false;
    }

    if( limit > 0 && limit < (int)count){
        count = limit;
    }

    //read label data
    auto label_data_buffer = reinterpret_cast<unsigned char*>(label_buffer.get() + 8);

    T* labels = new T[count * vec_size];
    memset(labels, 0, sizeof(T) * count * vec_size);

    for(size_t i = 0; i < count; i++){
        auto label = static_cast<size_t>(*label_data_buffer++);
        labels[i * vec_size + label] = 1;
    }

    out_mat->set_shallow_data(labels, count, vec_size);

    return true;
}

template<class T>
bool MnistHelper<T>::read_vec_labels(const std::string& label_file,
                                     abcdl::algebra::MatrixSet<T>& out_matrix_set,
                                     const int limit,
                                     const size_t vec_size){
    size_t count = 0;
    auto label_buffer = read_mnist_file(label_file, 0x801, &count);
    if(!label_buffer || count == 0){
        return false;
    }

    if( limit > 0 && limit < (int)count){
        count = limit;
    }

    //read label data
    auto label_data_buffer = reinterpret_cast<unsigned char*>(label_buffer.get() + 8);

    T* labels = new T[vec_size];
    for(size_t i = 0; i < count; i++){
        memset(labels, 0, sizeof(T) * vec_size);
        auto label = static_cast<size_t>(*label_data_buffer++);
        labels[label] = 1;
        out_matrix_set.push_back(abcdl::algebra::Matrix<T>(labels, 1, vec_size));
    }
    delete[] labels;

    return true;
}

template<class T>
inline std::unique_ptr<char[]> MnistHelper<T>::read_mnist_file(const std::string& path,
                                                               const int key,
                                                               size_t* out_rows){

    *out_rows = 0;

    std::ifstream file;
    file.open(path, std::ios::in | std::ios::binary | std::ios::ate);

    if(!file){
        std::cout << "Can't open file:" << path << std::endl;
        return {};
    }

    auto size = file.tellg();
    std::unique_ptr<char[]> buffer(new char[size]);

    //read the entire file to buffer
    file.seekg(0, std::ios::beg);
    file.read(buffer.get(), size);
    file.close();

    auto magic_num = read_header(buffer, 0);

    if(magic_num != key){
        std::cout << "Invalid magic number, probably not a MNIST file" << std::endl;
        return {};
    }

    auto count = read_header(buffer, 1);

    if(magic_num == 0x803){
        auto rows = read_header(buffer, 2);
        auto cols = read_header(buffer, 3);

        if(size < count * rows * cols + 16){
            std::cout << "File size ERROR" << std::endl;
            return {};
        }
    }else if( magic_num == 0x801){
        if(size < count + 8){
            std::cout << "File size ERROR" << std::endl;
            return {};
        }
    }

    *out_rows = count;

    return std::move(buffer);
}

template<class T>
inline int MnistHelper<T>::read_header(const std::unique_ptr<char[]>& buffer, size_t position){
    auto header = reinterpret_cast<int*>(buffer.get());
    auto value = *(header + position);
    return (value << 24) | ((value << 8) & 0x00FF0000) | ((value >> 8) & 0X0000FF00) | (value >> 24);
}

template<class T>
inline T* MnistHelper<T>::vectorize_label(const size_t data, const size_t sizes){
    T* vec = new T[sizes];
    memset(vec, static_cast<T>(0), sizeof(T) * sizes);

    if(data < sizes){
        vec[data] = static_cast<T>(1);
    }

    return vec;
}

}//namespace utils
}//namespace abcdl
