/*********************************************
* Author: Jun Jiang - jiangjun4@sina.com
* Created: 2017-08-24 15:03
* Last modified: 2017-08-31 20:21
* Filename: Pool.h
* Description: CNN pooling
**********************************************/
#pragma once

#include "algebra/Matrix.h"

namespace abcdl{
namespace framework{

class Pooling{
public:
    virtual ~Pooling()= default;
    virtual void pool(abcdl::algebra::Mat& pool,
                      const abcdl::algebra::Mat& mat,
                      const size_t rows,
                      const size_t cols,
                      const size_t scale) = 0;
};//class Pooling

class MeanPooling : public Pooling{
public:
    void pool(abcdl::algebra::Mat& pool,
              const abcdl::algebra::Mat& mat,
              const size_t rows,
              const size_t cols,
              const size_t scale);
};//class MeanPooling

class MaxPooling : public Pooling{
public:
    void pool(abcdl::algebra::Mat& pool,
              const abcdl::algebra::Mat& mat,
              const size_t rows,
              const size_t cols,
              const size_t scale);
};//class MaxPooling

class L2Pooling : public Pooling{
public:
    void pool(abcdl::algebra::Mat& pool,
              const abcdl::algebra::Mat& mat,
              const size_t rows,
              const size_t cols,
              const size_t scale);
};//class L2Pooling

}//namespace framework
}//namespace abcdl
