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
    Matrix(const std::size_t rows, const std::size_t cols);
    Matrix(const T& value,
           const std::size_t rows,
           const std::size_t cols);
    Matrix(const T* data,
           const std::size_t rows,
           const std::size_t cols);
    ~Matrix();

    inline std::size_t rows() const { return _rows;}
    inline std::size_t cols() const { return _cols;}
    inline std::size_t get_size() const { return _rows * _cols;}

    inline T* data() const { return _data; }
    inline T& get_data(const std::size_t idx) const;
    inline T& get_data(const std::size_t row_id, const std::size_t col_id) const;

    inline void set_data(const T& value, const std::size_t idx);
    inline void set_data(const T& value,
                         const std::size_t row_id,
                         const std::size_t col_id);
    inline void set_data(const T* data,
                         const std::size_t rows,
                         const std::size_t cols);
    inline void set_data(const Matrix<T>& mat);

    void set_shallow_data(T* data,
                          const std::size_t rows,
                          const std::size_t cols);
    
    Matrix<T>* get_row(const std::size_t row_id, const std::size_t row_size = 1);
    void get_row(Matrix<T>& mat, const std::size_t row_id, const std::size_t row_size = 1);
    void set_row(const Matrix<T>& mat);
    void set_row(const std::size_t row_id, const Matrix<T>& mat);
    void insert_row(const Matrix<T>& mat);
    void insert_row(const std::size_t row_id, const Matrix<T>& mat);

    Matrix<T>* get_col(const std::size_t col_id, const std::size_t col_size = 1);
	void get_col(Matrix<T>& mat, const std::size_t col_id, const std::size_t col_size = 1);
    void set_col(const Matrix<T>& mat);
    void set_col(const std::size_t col_id, const Matrix<T>& mat);
    void insert_col(const Matrix<T>& mat);
    void insert_col(const std::size_t col_id, const Matrix<T>& mat);

    Matrix<T>* clone() const;
    void clone(Matrix<T>& mat) const;

    void reset(const T& value = 0);
    void reset(const T& value,
               const std::size_t rows,
               const std::size_t cols);

    void display(const std::string& split="\t");

    //operator
	T& operator [] (const std::size_t idx) const;
	bool operator == (const Matrix<T>& mat) const;

    Matrix<T>& operator = (const T& value);
    Matrix<T>& operator = (const Matrix<T>& mat);

    Matrix<T> operator + (const T& value);
    Matrix<T> operator + (const Matrix<T>& mat);

    Matrix<T>& operator += (const T& value);
    Matrix<T>& operator += (const Matrix<T>& mat);

    Matrix<T> operator - (const T& value);
    Matrix<T> operator - (const Matrix<T>& mat);
    Matrix<T>& operator -= (const T& value);
    Matrix<T>& operator -= (const Matrix<T>& mat);

    Matrix<T> operator * (const T& value);
    Matrix<T> operator * (const Matrix<T>& mat);
    Matrix<T>& operator *= (const T& value);
    Matrix<T>& operator *= (const Matrix<T>& mat);

    Matrix<T> operator / (const T& value);
    Matrix<T> operator / (const Matrix<T>& mat);
    Matrix<T>& operator /= (const T& value);
    Matrix<T>& operator /= (const Matrix<T>& mat);

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

    Matrix<int>* argmax(const std::size_t axis);
    std::size_t argmax(const std::size_t id, const std::size_t axis);

	bool isnan();
	bool isinf();

    virtual bool swap(const std::size_t a_row,
                      const std::size_t a_col,
                      const std::size_t b_row,
                      const std::size_t b_col) = 0;
    virtual bool swap_row(const std::size_t a, const std::size_t b) = 0;
    virtual bool swap_col(const std::size_t a, const std::size_t b) = 0;

    virtual Matrix<T>* transpose() = 0;

    bool reshape(std::size_t row, std::size_t col){
        if(row * col == _rows * _cols){
            _rows = row;
            _cols = col;
            return true;
        }
        return false;
    }

    void expand(std::size_t row_dim, std::size_t col_dim);

    //shape is full or valid, default full.
    bool convn(ccma::algebra::Matrix<T>* kernal,
               std::size_t stride = 1,
               std::string shape = "full");

    
    //convert dim 1 is row dim, 2 is col dim
    void flipdim(std::size_t dim = 1);
    void flip180();

    virtual void add_x0() = 0;
    virtual void add_x0(Matrix<T>* result) = 0;

    virtual bool det(T* result) = 0;

    virtual real mean() = 0;
    virtual real mean(std::size_t col) = 0;

    virtual real var() = 0;
    virtual real var(std::size_t col) = 0;

    virtual bool inverse(Matrix<real>* result) = 0;
*/

private:
    inline bool equal_shape(const Matrix<T>& mat) const{
        return _rows == mat.rows() && _cols == mat.cols();
    }
private:
    std::size_t _rows;
    std::size_t _cols;
    T*   _data;
};//class Matrix

}//namespace algebra
}//namespace abcdl

#endif //_ABCDL_ALGEBRA_MATRIX_H_
