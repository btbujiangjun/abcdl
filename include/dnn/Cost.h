/***********************************************
 * Author: Jun Jiang - jiangjun4@sina.com
 * Create: 2017-08-18 13:25
 * Last modified : 2017-08-18 13:25
 * Filename      : Cost.h
 * Description   : Cost function 
 **********************************************/

#pragma once

#include "algebra/Matrix.h"
#include "algebra/MatrixHelper.h"

namespace abcdl{
namespace dnn{

class Cost{
public:
    virtual ~Cost(){}
    virtual void delta(abcdl::algebra::Mat mat,
                       const abcdl::algebra::Mat& activate,
                       const abcdl::algebra::Mat& y);
protected:
    abcdl::algebra::MatrixHelper<real> helper;
};//class Cost

/*
 * Cost = (y - a) ^2 / 2
 */
class QuadraticCost : public Cost{
public:
    ~QuadraticCost(){}
    void delta(abcdl::algebra::Mat mat,
               const abcdl::algebra::Mat& activate,
               const abcdl::algebra::Mat& y){
        abcdl::algebra::Mat derivative_mat;
        helper.sigmoid_derivative(derivative_mat, activate);
        mat = (activate - y) * derivative_mat;
    }
};//class QuadraticCost

/*
 * derivative C_w = 1/n * ∑(x_j(σ(z) - y))
 */
class CrossEntropyCost : public Cost{
    ~CrossEntropyCost(){}
    void delta(abcdl::algebra::Mat mat,
               const abcdl::algebra::Mat& activate,
               const abcdl::algebra::Mat& y){
        mat = activate - y;
    }
};//class CrossEntropyCost

}//namespace dnn
}//namespace abcdl
