/***********************************************
 * Author: Jun Jiang - jiangjun4@sina.com
 * Create: 2017-08-17 19:30
 * Last modified : 2017-08-17 19:30
 * Filename      : ActivateFunc.h
 * Description   : 
 **********************************************/

#pragma once

#include "algebra/Matrix.h"
#include "algebra/MatrixHelper.h"

namespace abcdl{
namespace dnn{

class ActivateFunc{
public:
    virtual ~ActivateFunc() = default;
    virtual void activate(abcdl::algebra::Mat mat, const abcdl::algebra::Mat& z_mat) = 0;
    virtual void derivative(abcdl::algebra::Mat mat, const abcdl::algebra::Mat& activate_mat) = 0;
protected:
    abcdl::algebra::MatrixHelper<real> helper;
};//class ActivateFunc

class SigmoidActivateFunc : public ActivateFunc{
public:
    void activate(abcdl::algebra::Mat mat, const abcdl::algebra::Mat& z_mat){
        helper.sigmoid(mat, z_mat);
    }
    void derivative(abcdl::algebra::Mat mat, const abcdl::algebra::Mat& activate_mat){
        helper.sigmoid_derivative(mat, activate_mat);
    }
};//class SigmoidActivateFunc

class ReluActivateFunc : public ActivateFunc{
    void activate(abcdl::algebra::Mat mat, const abcdl::algebra::Mat& z_mat){
        helper.relu(mat, z_mat);
    }
    void derivative(abcdl::algebra::Mat mat, const abcdl::algebra::Mat& activate_mat){
    }
};//class ReluActivateFunc

}//namespace dnn
}//namespace abcdl
