/***********************************************
 * Author: Jun Jiang - jiangjun4@sina.com
 * Create: 2017-08-17 19:30
 * Last modified : 2017-09-01 10:15
 * Filename      : ActivateFunc.h
 * Description   : Activate Function 
 **********************************************/
#pragma once

#include "algebra/Matrix.h"
#include "algebra/MatrixHelper.h"
#include "framework/Cost.h"

namespace abcdl{
namespace framework{

class ActivateFunc{
public:
    virtual ~ActivateFunc() = default;
    virtual void activate(abcdl::algebra::Mat& mat, const abcdl::algebra::Mat& z_mat) = 0;
    virtual void derivative(abcdl::algebra::Mat& mat, const abcdl::algebra::Mat& activate_mat) = 0;
protected:
    abcdl::algebra::MatrixHelper<real> helper;
};//class ActivateFunc

class SigmoidActivateFunc : public ActivateFunc{
public:
    void activate(abcdl::algebra::Mat& mat, const abcdl::algebra::Mat& z_mat){
        helper.sigmoid(mat, z_mat);
    }
    void derivative(abcdl::algebra::Mat& mat, const abcdl::algebra::Mat& activate_mat){
        helper.sigmoid_derivative(mat, activate_mat);
    }
};//class SigmoidActivateFunc

class TanhActivateFunc : public ActivateFunc{
    void activate(abcdl::algebra::Mat& mat, const abcdl::algebra::Mat& z_mat){
        helper.tanh(mat, z_mat);
    }
    void derivative(abcdl::algebra::Mat& mat, const abcdl::algebra::Mat& activate_mat){
        helper.tanh_derivative(mat, activate_mat);
    }
};//class TanhActivateFunc

class ReluActivateFunc : public ActivateFunc{
    void activate(abcdl::algebra::Mat& mat, const abcdl::algebra::Mat& z_mat){
        helper.relu(mat, z_mat);
    }
    void derivative(abcdl::algebra::Mat& mat, const abcdl::algebra::Mat& activate_mat){
        helper.relu_derivative(mat, activate_mat);
    }
};//class ReluActivateFunc

class LeakyReluActivateFunc : public ActivateFunc{
    void activate(abcdl::algebra::Mat& mat, const abcdl::algebra::Mat& z_mat){
        helper.leaky_relu(mat, z_mat);
    }
    void derivative(abcdl::algebra::Mat& mat, const abcdl::algebra::Mat& activate_mat){
        helper.leaky_relu_derivative(mat, activate_mat);
    }
};//class TanhActivateFunc

class EluActivateFunc : public ActivateFunc{
    void activate(abcdl::algebra::Mat& mat, const abcdl::algebra::Mat& z_mat){
        helper.elu(mat, z_mat);
    }
    void derivative(abcdl::algebra::Mat& mat, const abcdl::algebra::Mat& activate_mat){
        helper.elu_derivative(mat, activate_mat);
    }
};//class EluActivateFunc

}//namespace framework
}//namespace abcdl
