/**********************************************
* Author: Jun Jiang - jiangjun4@sina.com
* Created: 2017-07-25 15:19
* Last modified: 2017-07-25 15:19
* Filename: Matrix.h
* Description: Base Data Structure
**********************************************/
#pragma once

#include <string>
#include <cstring>
#include "utils/TypeDef.h"
#include "utils/Log.h"


namespace abcdl{
namespace algebra{

template<class T>
class Matrix{
public:
    Matrix();
    Matrix(const size_t rows, const size_t cols);
    Matrix(const T& value,
           const size_t rows,
           const size_t cols);
    Matrix(const T* data,
           const size_t rows,
           const size_t cols);
    ~Matrix();

    inline size_t rows() const { return _rows;}
    inline size_t cols() const { return _cols;}
    inline size_t get_size() const { return _rows * _cols;}

    inline T* data() const { return _data; }
    inline T& get_data(const size_t idx) const{
        CHECK(idx < get_size());
		return _data[idx];
	}
    inline T& get_data(const size_t row_id, const size_t col_id) const{
        size_t size = row_id * _cols + col_id;
        CHECK(size < get_size());
		return _data[size];
	}

    inline void set_data(const T& value, const size_t idx){
        CHECK(idx < get_size());
		_data[idx] = value;
	}
    inline void set_data(const T& value,
                         const size_t row_id,
                         const size_t col_id){
        size_t size = row_id * _cols + col_id;
        CHECK(size < get_size());
		_data[size] = value;
	}
    inline void set_data(const T* data,
                         const size_t rows,
                         const size_t cols){
        if(_rows * _cols != rows * cols){
            if(_data != nullptr){
                delete[] _data;
            }
    		_data = new T[rows * cols];
        }
		_rows = rows;
		_cols = cols;
		memcpy(_data, data, sizeof(T) * rows * cols);
	}
    inline void set_data(const Matrix<T>& mat){
		set_data(mat.data(), mat.rows(), mat.cols());
	}

    void set_shallow_data(T* data,
                          const size_t rows,
                          const size_t cols);
    
    Matrix<T> get_row(const size_t row_id, const size_t row_size = 1) const;
    void get_row(Matrix<T>* mat,
				 const size_t row_id,
				 const size_t row_size = 1) const;
    void set_row(const Matrix<T>& mat);
    void set_row(const size_t row_id, const Matrix<T>& mat);
    void insert_row(const Matrix<T>& mat);
    void insert_row(const size_t row_id, const Matrix<T>& mat);

    Matrix<T> get_col(const size_t col_id, const size_t col_size = 1) const;
	void get_col(Matrix<T>* mat,
				 const size_t col_id,
				 const size_t col_size = 1) const;
    void set_col(const Matrix<T>& mat);
    void set_col(const size_t col_id, const Matrix<T>& mat);
    void insert_col(const Matrix<T>& mat);
    void insert_col(const size_t col_id, const Matrix<T>& mat);

    Matrix<T>* clone() const;
    void clone(Matrix<T>& mat) const;

    void reset(const T& value = 0);
    void reset(const T& value,
               const size_t rows,
               const size_t cols);

    void transpose();
    Matrix<T> Ts();

    inline bool reshape(size_t row, size_t col){
        if(row * col == _rows * _cols){
            _rows = row;
            _cols = col;
            return true;
        }
        return false;
    }

    void display(const std::string& split="\t");

    //operator
	T& operator [] (const size_t idx) const;
	bool operator == (const Matrix<T>& mat) const;

    Matrix<T>& operator = (const T& value);
    Matrix<T>& operator = (const Matrix<T>& mat);

    Matrix<T> operator + (const T& value) const;
    Matrix<T> operator + (const Matrix<T>& mat) const;

    Matrix<T>& operator += (const T& value);
    Matrix<T>& operator += (const Matrix<T>& mat);

    Matrix<T> operator - (const T& value) const;
    Matrix<T> operator - (const Matrix<T>& mat) const;
    Matrix<T>& operator -= (const T& value);
    Matrix<T>& operator -= (const Matrix<T>& mat);

    Matrix<T> operator * (const T& value) const;
    Matrix<T> operator * (const Matrix<T>& mat) const;
    Matrix<T>& operator *= (const T& value);
    Matrix<T>& operator *= (const Matrix<T>& mat);

    Matrix<T> operator / (const T& value) const;
    Matrix<T> operator / (const Matrix<T>& mat) const;
    Matrix<T>& operator /= (const T& value);
    Matrix<T>& operator /= (const Matrix<T>& mat);

    //algebra
    void dot(const Matrix<T>& mat);
    void outer(const Matrix<T>& mat);
    void pow(const T& exponent);
    void log();
    void exp();
    void sigmoid();
	void softmax();
	void tanh();
	void relu();


	T max() const;
	T min() const;
    T sum() const;
    real mean() const;
    bool inverse(Matrix<real>& mat);

//    bool det(T* result);
/*

    virtual T sum() const = 0;
    void x_sum();
    void y_sum();

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


    void expand(size_t row_dim, size_t col_dim);

    //shape is full or valid, default full.
    bool convn(ccma::algebra::Matrix<T>* kernal,
               size_t stride = 1,
               std::string shape = "full");

    
    //convert dim 1 is row dim, 2 is col dim
    void flipdim(size_t dim = 1);
    void flip180();

    virtual void add_x0() = 0;
    virtual void add_x0(Matrix<T>* result) = 0;

    virtual bool det(T* result) = 0;

    virtual real mean(size_t col) = 0;

    virtual real var() = 0;
    virtual real var(size_t col) = 0;

    virtual bool inverse(Matrix<real>* result) = 0;
*/

private:
    inline bool equal_shape(const Matrix<T>& mat) const{
        return _rows == mat.rows() && _cols == mat.cols();
    }
protected:
    size_t _rows;
    size_t _cols;
    T*   _data;
};//class Matrix


template<class T>
class RandomMatrix : public Matrix<T>{
public:
    RandomMatrix(){}
    RandomMatrix(size_t rows,
                 size_t cols,
                 const T& mean_value,
                 const T& stddev,
                 const T& min = 0,
                 const T& max = 0);
    void reset();
    void reset(size_t rows,
               size_t cols,
               const T& mean_value,
               const T& stddev,
               const T& min = 0,
               const T& max = 0);
private:
    T _mean_value;
    T _stddev;
    T _min;
    T _max;
};//class RandomMatrix

template<class T>
class EyeMatrix : Matrix<T>{
    explicit EyeMatrix(size_t size){
        T* data = new T[size * size];
        memset(data, 0, sizeof(T) * size * size);
        for(size_t i = 0; i != size; i++){
            data[i * size + i] = static_cast<T>(1);
        }
        this->set_shallow_data(data, size, size);
    }
};//class EyeMatrix

typedef Matrix<real> Mat;
typedef Matrix<int> IMat;

}//namespace algebra
}//namespace abcdl
