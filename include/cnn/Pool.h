/*********************************************
* Author: Jun Jiang - jiangjun4@sina.com
* Created: 2017-08-24 15:03
* Last modified: 2017-08-24 19:30
* Filename: Layer.h
* Description: CNN pooling layer
**********************************************/
#pragma once

#include <vector>
#include "algebra/Matrix.h"

namespace ccma{
namespace cnn{

class Pooling{
public:
    virtual ~Pooling(){}
    virtual void pool(abcdl::algebra::Mat& pool,
                      const size_t rows,
                      const size_t cols,
                      const size_t scale,
                      const abcdl::algebra::Mat& mat);
};//class Pooling

class MeanPooling : public Pooling{
public:
    void pool(abcdl::algebra::Mat& pool,
              const size_t rows,
              const size_t cols,
              const size_t scale,
              const abcdl::algebra::Mat& mat);
};//class MeanPooling

class MaxPooling : public Pooling{
public:
    void pool(abcdl::algebra::Mat& pool,
              const size_t rows,
              const size_t cols,
              const size_t scale,
              const abcdl::algebra::Mat& mat);
};//class MaxPooling

class L2Pooling : public Pooling{
public:
    void pool(abcdl::algebra::Mat& pool,
              const size_t rows,
              const size_t cols,
              const size_t scale,
              const abcdl::algebra::Mat& mat);
};//class L2Pooling

}//namespace cnn
}//namespace ccma