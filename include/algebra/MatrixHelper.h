/**********************************************
* Author: Jun Jiang - jiangjun4@sina.com
* Created: 2017-07-25 15:19
* Last modified: 2017-07-27 16:53
* Filename: MatrixHelper.h
* Description: matrix common operation
**********************************************/

#ifndef _ABCDL_ALGEBRA_MATRIXHELPER_H_
#define _ABCDL_ALGEBRA_MATRIXHELPER_H_

#include "algebra/Matrix.h"
#include "utils/ParallelOperator.h"

using abcdl::algebra::Matrix;

namespace abcdl{
namespace algebra{

template<class T>
class MatrixHelper{
public:
    //algebra
    Matrix<T> dot(const Matrix<T>& mat_a,
                  const Matrix<T>& mat_b);
    void dot(Matrix<T>& mat,
			 const Matrix<T>& mat_a,
			 const Matrix<T>& mat_b);
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
	void relu(Matrix<T>& mat, const Matrix<T>& mat_a);


	void transpose(Matrix<T>& mat, const Matrix<T>& mat_a);

/*
    Matrix<int>* argmax(const size_t axis);
    size_t argmax(const size_t id, const size_t axis);

	bool isnan();
	bool isinf();

    virtual bool swap(const size_t a_row,
                      const size_t a_col,
                      const size_t b_row,
                      const size_t b_col) = 0;
    virtual bool swap_row(const size_t a, const size_t b) = 0;
    virtual bool swap_col(const size_t a, const size_t b) = 0;

    virtual Matrix<T>* transpose() = 0;

    bool reshape(size_t row, size_t col){
        if(row * col == _rows * _cols){
            _rows = row;
            _cols = col;
            return true;
        }
        return false;
    }

    Matrix<T> expand(size_t row_dim, size_t col_dim);

    //shape is full or valid, default full.
    bool convn(ccma::algebra::Matrix<T>* kernal,
               size_t stride = 1,
               std::string shape = "full");

    
    //convert dim 1 is row dim, 2 is col dim
    Matrix<T> flipdim(size_t dim = 1);
    Matrix<T> flip180();

    virtual Matrix<T> add_x0() = 0;
    virtual Matrix<T> add_x0(Matrix<T>* result) = 0;

    virtual bool det(T* result) = 0;

    virtual real mean() = 0;
    virtual real mean(size_t col) = 0;

    virtual real var() = 0;
    virtual real var(size_t col) = 0;

    virtual bool inverse(Matrix<real>* result) = 0;
*/

    void zero_like(Matrix<T>& mat, const Matrix<T>& mat_a);

private:
    abcdl::utils::ParallelOperator po;

};//class MatrixHelper

}//namespace algebra
}//namespace abcdl

#endif //_ABCDL_ALGEBRA_MATRIXHELPER_H_
