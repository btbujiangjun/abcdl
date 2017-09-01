/*********************************************
* Author: Jun Jiang - jiangjun4@sina.com
* Created: 2017-04-18 17:41
* Last modified: 2017-09-01 10:53
* Filename: MnistHelper.h
* Description: read mnist file
**********************************************/

#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include "algebra/Matrix.h"

namespace abcdl{
namespace utils{

template<class T>
class MnistHelper{
public:
    bool read_image(const std::string& image_file,
                    abcdl::algebra::Matrix<T>* out_mat,
                    const int limit = -1,
                    const uint threshold = 30);

    bool read_label(const std::string& label_file,
                    abcdl::algebra::Matrix<T>* out_mat,
                    const int limit = -1);

    bool read_vec_label(const std::string& label_file,
                        abcdl::algebra::Matrix<T>* out_mat,
                        const int limit = -1,
                        const uint vec_size = 10);

private:
    inline std::unique_ptr<char[]> read_mnist_file(const std::string& path,
                                                   uint key,
                                                   uint* out_rows);
    inline uint read_header(const std::unique_ptr<char[]>& buffer, size_t position);
    inline T* vectorize_label(const uint data, const uint sizes);

};//class MnistHelper


template<class T>
bool MnistHelper<T>::read_image(const std::string& image_file,
                                abcdl::algebra::Matrix<T>* out_mat,
                                const int limit,
                                const uint threshold){

    uint count = 0;
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

    uint size = count * rows * cols;
    T* data = new T[size];
    for(size_t i = 0; i < size; i++){
        data[i] = static_cast<T>(*image_data_buffer++) > threshold ? 1 : 0;
    }

    out_mat->set_shallow_data(data, count, rows * cols);

    return true;
}

template<class T>
bool MnistHelper<T>::read_label(const std::string& label_file,
                                abcdl::algebra::Matrix<T>* out_mat,
                                const int limit){
    uint count = 0;
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
                                const uint vec_size){
    uint count = 0;
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
        auto label = static_cast<uint>(*label_data_buffer++);
        labels[i * vec_size + label] = 1;
    }

    out_mat->set_shallow_data(labels, count, vec_size);

    return true;
}
template<class T>
inline std::unique_ptr<char[]> MnistHelper<T>::read_mnist_file(const std::string& path,
                                                               uint key,
                                                               uint* out_rows){

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
inline uint MnistHelper<T>::read_header(const std::unique_ptr<char[]>& buffer, size_t position){
    auto header = reinterpret_cast<uint*>(buffer.get());
    auto value = *(header + position);
    return (value << 24) | ((value << 8) & 0x00FF0000) | ((value >> 8) & 0X0000FF00) | (value >> 24);
}

template<class T>
inline T* MnistHelper<T>::vectorize_label(const uint data, const uint sizes){
    T* vec = new T[sizes];
    memset(vec, static_cast<T>(0), sizeof(T) * sizes);

    if(data < sizes){
        vec[data] = static_cast<T>(1);
    }

    return vec;
}

}//namespace utils
}//namespace abcdl
