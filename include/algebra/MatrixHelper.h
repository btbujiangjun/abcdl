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
    Matrix<T> dot(const Matrix<T>& mat_a, const Matrix<T>& mat_b);
    Matrix<T> outer(const Matrix<T>& mat_a, const Matrix<T>& mat_b);
    Matrix<T> pow(const Matrix<T>& mat, const T& exponent);
    Matrix<T> log(const Matrix<T>& mat);
    Matrix<T> exp(const Matrix<T>& mat);
    Matrix<T> sigmoid(const Matrix<T>& mat);
	Matrix<T> softmax(const Matrix<T>& mat);
	Matrix<T> tanh(const Matrix<T>& mat);
	Matrix<T> relu(const Matrix<T>& mat);

/*
    Matrix<int>* argmax(const uint axis);
    uint argmax(const uint id, const uint axis);

	bool isnan();
	bool isinf();

    virtual bool swap(const uint a_row,
                      const uint a_col,
                      const uint b_row,
                      const uint b_col) = 0;
    virtual bool swap_row(const uint a, const uint b) = 0;
    virtual bool swap_col(const uint a, const uint b) = 0;

    virtual Matrix<T>* transpose() = 0;

    bool reshape(uint row, uint col){
        if(row * col == _rows * _cols){
            _rows = row;
            _cols = col;
            return true;
        }
        return false;
    }

    Matrix<T> expand(uint row_dim, uint col_dim);

    //shape is full or valid, default full.
    bool convn(ccma::algebra::Matrix<T>* kernal,
               uint stride = 1,
               std::string shape = "full");

    
    //convert dim 1 is row dim, 2 is col dim
    Matrix<T> flipdim(uint dim = 1);
    Matrix<T> flip180();

    virtual Matrix<T> add_x0() = 0;
    virtual Matrix<T> add_x0(Matrix<T>* result) = 0;

    virtual bool det(T* result) = 0;

    virtual real mean() = 0;
    virtual real mean(uint col) = 0;

    virtual real var() = 0;
    virtual real var(uint col) = 0;

    virtual bool inverse(Matrix<real>* result) = 0;
*/

    Matrix<T> zero_like(const Matrix<T>& mat);

    friend class Matrix<T>;
private:
    void dot(const Matrix<T>& mat_a, const Matrix<T>& mat_b, Matrix<T>& mat);
    void outer(const Matrix<T>& mat_a, const Matrix<T>& mat_b, Matrix<T>& mat);
    void pow(const Matrix<T>& mat_a, const T& exponent, Matrix<T>& mat);
    void log(const Matrix<T>& mat_a, Matrix<T>& mat);
    void exp(const Matrix<T>& mat_a, Matrix<T>& mat);
    void sigmoid(const Matrix<T>& mat_a, Matrix<T>& mat);
	void softmax(const Matrix<T>& mat_a, Matrix<T>& mat);
	void tanh(const Matrix<T>& mat_a, Matrix<T>& mat);
	void relu(const Matrix<T>& mat_a, Matrix<T>& mat);

private:
    abcdl::utils::ParallelOperator po;

};//class MatrixHelper

}//namespace algebra
}//namespace abcdl

#endif //_ABCDL_ALGEBRA_MATRIXHELPER_H_
