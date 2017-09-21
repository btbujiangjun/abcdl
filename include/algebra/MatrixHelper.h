/**********************************************
* Author: Jun Jiang - jiangjun4@sina.com
* Created: 2017-07-25 15:19
* Last modified: 2017-07-27 16:53
* Filename: MatrixHelper.h
* Description: matrix common operation
**********************************************/
#pragma once

#include "algebra/Matrix.h"
#include "utils/ParallelOperator.h"

using abcdl::algebra::Matrix;

namespace abcdl{
namespace algebra{

template<class T>
class MatrixHelper{
public:
    Matrix<T> dot(const Matrix<T>& mat_a, const Matrix<T>& mat_b);
    void dot(Matrix<T>& mat,
			 const Matrix<T>& mat_a,
			 const Matrix<T>& mat_b);
    Matrix<T> outer(const Matrix<T>& mat_a, const Matrix<T>& mat_b);
    void outer(Matrix<T>& mat,
			   const Matrix<T>& mat_a,
			   const Matrix<T>& mat_b);
    void pow(Matrix<T>& mat,
			 const Matrix<T>& mat_a,
			 const T& exponent);
    void log(Matrix<T>& mat,
			 const Matrix<T>& mat_a);
    void exp(Matrix<T>& mat,
			 const Matrix<T>& mat_a);
    void sigmoid(Matrix<T>& mat, const Matrix<T>& mat_a);
    void sigmoid_derivative(Matrix<T>& mat, const Matrix<T>& mat_a);
	void softmax(Matrix<T>& mat, const Matrix<T>& mat_a);
	void tanh(Matrix<T>& mat, const Matrix<T>& mat_a);
	void tanh_derivative(Matrix<T>& mat, const Matrix<T>& mat_a);
	void relu(Matrix<T>& mat, const Matrix<T>& mat_a);
	void relu_derivative(Matrix<T>& mat, const Matrix<T>& mat_a);
	void leaky_relu(Matrix<T>& mat, const Matrix<T>& mat_a);
	void leaky_relu_derivative(Matrix<T>& mat, const Matrix<T>& mat_a);
	void elu(Matrix<T>& mat, const Matrix<T>& mat_a);
	void elu_derivative(Matrix<T>& mat, const Matrix<T>& mat_a);
    void expand(Matrix<T>& result,
                const Matrix<T>& mat,
                const size_t row_dim,
                const size_t col_dim);

    bool convn(Matrix<T>& result,
               const Matrix<T>& mat,
               const Matrix<T>& kernal,
               const size_t stride,
               const Convn_type type = VALID);

	void transpose(Matrix<T>& mat, const Matrix<T>& mat_a);

    void zero_like(Matrix<T>& mat, const Matrix<T>& mat_a);

private:
    abcdl::utils::ParallelOperator po;

};//class MatrixHelper

}//namespace algebra
}//namespace abcdl
