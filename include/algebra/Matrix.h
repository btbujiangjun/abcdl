/**********************************************
* Author: Jun Jiang - jiangjun4@sina.com
* Created: 2017-07-25 15:19
* Last modified: 2017-07-25 15:19
* Filename: Matrix.h
* Description: Base Data Structure
**********************************************/

#ifndef _ABCDL_ALGEBRA_MATRIX_H_
#define _ABCDL_ALGEBRA_MATRIX_H_

#include <string>
#include "utils/TypeDef.h"

namespace abcdl{
namespace algebra{

template<class T>
class Matrix{
public:
    Matrix();
    Matrix(const uint rows, const uint cols);
    Matrix(const T& value,
           const uint rows,
           const uint cols);
    Matrix(const T* data,
           const uint rows,
           const uint cols);
    ~Matrix();

    inline uint rows() const { return _rows;}
    inline uint cols() const { return _cols;}
    inline uint get_size() const { return _rows * _cols;}

    inline T* data() const { return _data; }
    inline T& get_data(const uint idx) const;
    inline T& get_data(const uint row_id, const uint col_id) const;

    inline void set_data(const T& value, const uint idx);
    inline void set_data(const T& value,
                         const uint row_id,
                         const uint col_id);
    inline void set_data(const T* data,
                         const uint rows,
                         const uint cols);
    inline void set_data(const Matrix<T>& mat);

    void set_shallow_data(T* data,
                          const uint rows,
                          const uint cols);
    
    Matrix<T> get_row(const uint row_id, const uint row_size = 1);
    void set_row(const Matrix<T>& mat);
    void set_row(const uint row_id, const Matrix<T> mat);
    void insert_row(const Matrix<T>& mat);
    void insert_row(const uint row_id, const Matrix<T>& mat);

    Matrix<T> get_col(const uint col_id, const uint col_size = 1);
    void set_col(const Matrix<T>& mat);
    void set_col(const uint col_id, const Matrix<T>& mat);
    void insert_col(const Matrix<T>& mat);
    void insert_col(const uint col_id, const Matrix<T>& mat);

    Matrix<T> clone() const;

    void reset(const T& value = 0);
    void reset(const T& value,
               const uint rows,
               const uint cols);

    void display(const std::string& split="\t");

    //operator
    bool operator == (const Matrix<T>& mat) const;

    Matrix<T>& operator = (const T& value);
    Matrix<T>& operator = (const Matrix<T>& mat);

    Matrix<T> operator + (const T& value);
    Matrix<T> operator + (const Matrix<T>& mat);

    Matrix<T>& operator += (const T& value);
    Matrix<T>& operator += (const Matrix<T>& mat);

    Matrix<T> operator - (const T& value);
    Matrix<T> operator - (const Matrix<T>& mat);
    void operator -= (const T& value);
    void operator -= (const Matrix<T>& mat);

    Matrix<T> operator * (const T& value);
    Matrix<T> operator * (const Matrix<T>& mat);
    void operator *= (const T& value);
    void operator *= (const Matrix<T>& mat);

    Matrix<T> operator / (const T& value);
    Matrix<T> operator / (const Matrix<T>& mat);
    void operator /= (const T& value);
    void operator /= (const Matrix<T>& mat);

    //algebra
    void dot(const Matrix<T>& mat);
    void outer(const Matrix<T>& mat);
    void pow(const T& exponent);
    void log();
    void exp();
    void sigmoid();
    //void derivative_sigmoid();
	void softmax();
	void tanh();
	void relu();

/*
    void pow(const T exponent);
    void log();
    void exp();
    void sigmoid();
    void derivative_sigmoid();
	void softmax();
	void tanh();
	void relu();

    virtual T sum() const = 0;
    void x_sum();
    void y_sum();

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

    void expand(uint row_dim, uint col_dim);

    //shape is full or valid, default full.
    bool convn(ccma::algebra::Matrix<T>* kernal,
               uint stride = 1,
               std::string shape = "full");

    
    //convert dim 1 is row dim, 2 is col dim
    void flipdim(uint dim = 1);
    void flip180();

    virtual void add_x0() = 0;
    virtual void add_x0(Matrix<T>* result) = 0;

    virtual bool det(T* result) = 0;

    virtual real mean() = 0;
    virtual real mean(uint col) = 0;

    virtual real var() = 0;
    virtual real var(uint col) = 0;

    virtual bool inverse(Matrix<real>* result) = 0;
*/

private:
    inline bool equal_shape(const Matrix<T>& mat) const{
        return _rows == mat.rows() && _cols == mat.cols();
    }
private:
    uint _rows;
    uint _cols;
    T*   _data;
};//class Matrix

}//namespace algebra
}//namespace abcdl

#endif //_ABCDL_ALGEBRA_MATRIX_H_
