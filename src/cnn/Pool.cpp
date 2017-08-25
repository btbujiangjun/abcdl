/*********************************************
* Author: Jun Jiang - jiangjun4@sina.com
* Created: 2017-08-24 16:47
* Last modified: 2017-08-24 19:39
* Filename: Pool.cpp
* Description: convolutional network pooling 
**********************************************/
#include <typeinfo>
#include <math.h>
#include "algorithm/cnn/Layer.h"

namespace abcdl{
namespace cnn{

void MeanPooling::pool(abcdl::algebra::Mat& pool,
                       const size_t rows,
                       const size_t cols,
                       const size_t scale,
                       const abcdl::algebra::Mat& mat){
    real* data = new real[rows * cols];
    size_t pooling_size = scale * scale;
    for(size_t j = 0; j != rows; j++){
        for(size_t k = 0; k != cols; k++){
            real pooling_value = 0;
            for(size_t m = 0; m != scale; m++){
                for(size_t n = 0; n != scale; n++){
                    pooling_value += mat.get_data(j * scale + m, k * scale + n);
                }
            }
            data[j * cols + k] = pooling_value / pooling_size;
        }
    }
    pool.set_shallow_data(data, rows, cols);
}
void MaxPooling::pool(abcdl::algebra::Mat& pool,
                      const size_t rows,
                      const size_t cols,
                      const size_t scale,
                      const abcdl::algebra::Mat& mat){
    real* data = new real[rows * cols];
    for(size_t j = 0; j != rows; j++){
        for(size_t k = 0; k != cols; k++){
            real pooling_value = 0;
            for(size_t m = 0; m != scale; m++){
                for(size_t n = 0; n != scale; n++){
                    real value = mat.get_data(j * scale + m, k * scale + n);
                    if((m == 0 && n == 0) || value > pooling_value){
                        pooling_value = value;
                    }
                }
            }
            data[j * cols + k] = pooling_value;
        }
    }
    pool.set_shallow_data(data, rows, cols);
}
void L2Pooling::pool(abcdl::algebra::Mat& pool,
                     const size_t rows,
                     const size_t cols,
                     const size_t scale,
                     const abcdl::algebra::Mat& mat){
    real* data = new real[rows * cols];
    for(size_t j = 0; j != rows; j++){
        for(size_t k = 0; k != cols; k++){
            real pooling_value = 0;
            for(size_t m = 0; m != scale; m++){
                for(size_t n = 0; n != scale; n++){
                    real value = mat.get_data(j * scale + m, k * scale + n);
                    if(value != 0){
                        pooling_value += (value * value);
                    }
                }
            }
            data[j * cols + k] = sqrt(pooling_value);
        }
    }
    pool.set_shallow_data(data, rows, cols);
}

}//namespace cnn
}//namespace abcdl
