/***********************************************
 * Author: Jun Jiang - jiangjun4@sina.com
 * Create: 2017-07-25 16:33
 * Last modified : 2017-07-25 16:33
 * Filename      : Matrix.h
 * Description   : Base Data Structure
 **********************************************/

#include "algebra/Matrix.h"
#include "utils/ParallelOperator.h"
#include <string.h>
#include <iostream>
#include <stdio.h>

using abcdl::utils::ParallelOperator;

namespace abcdl{
namespace algebra{

template<class T>
Matrix<T>::Matrix(){
    _rows = 0;
    _cols = 0;
    _data = nullptr;
}

template<class T>
Matrix<T>::Matrix(const std::size_t rows, const std::size_t cols){
    _rows = rows;
    _cols = cols;
    _data = new T[_rows * _cols];
    memset(_data, 0, sizeof(T) * _rows * _cols);
}

template<class T>
Matrix<T>::Matrix(const T& value,
                  const std::size_t rows,
                  const std::size_t cols){
    _rows = rows;
    _cols = cols;
    _data = new T[_rows * _cols];
    if(value == 0 || value == -1){
        memset(_data, value, sizeof(T) * _rows * _cols);
    }else{
        auto lamda = [](T* a, const T& b){ *a = b; };
        ParallelOperator po;
        po.parallel_mul2one<T>(_data, get_size(), value, lamda);
    }
}

template<class T>
Matrix<T>::Matrix(const T* data,
                  const std::size_t rows,
                  const std::size_t cols){
    _rows = rows;
    _cols = cols;
    _data = new T[_rows * _cols];
    memcpy(_data, data, sizeof(T) * _rows * _cols);
}

template<class T>
Matrix<T>::~Matrix(){
    if(_data != nullptr){
        printf("~Matrix.\n");
        delete[] _data;
        printf("~Finished Matrix.\n");
        _data = nullptr;
    }
}

template<class T>
T& Matrix<T>::get_data(const std::size_t idx) const{
    return _data[idx];
}

template<class T>
T& Matrix<T>::get_data(const std::size_t row_id, const std::size_t col_id) const{
    return _data[row_id * _cols + col_id];
}

template<class T>
void Matrix<T>::set_data(const T& value, const std::size_t idx){
    _data[idx] = value;
}

template<class T>
void Matrix<T>::set_data(const T& value,
                         const std::size_t row_id,
                         const std::size_t col_id){
    _data[row_id * _cols + col_id] = value; 
}

template<class T>
void Matrix<T>::set_data(const T* data,
                         const std::size_t rows,
                         const std::size_t cols){
    std::size_t size = rows * cols;
    if(_rows * _cols != size){
        if(_data != nullptr){
            delete[] _data;
            _data = nullptr;
        }
        _data = new T[size];
    }

    _rows = rows;
    _cols = cols;
    memcpy(_data, data, sizeof(T) * size);
}

template<class T>
void Matrix<T>::set_data(const Matrix<T>& mat){
    set_data(mat.data(), mat.rows(), mat.cols());
}

template<class T>
void Matrix<T>::set_shallow_data(T* data,
                                 const std::size_t rows,
                                 const std::size_t cols){
    if(_data != nullptr){
        delete[] _data;
        _data = nullptr;
    }
    _data = data;
    _rows = rows;
    _cols = cols;
}

template<class T>
Matrix<T> Matrix<T>::get_row(const std::size_t row_id, const std::size_t row_size){
    if(row_id + row_size > _rows){
        //todo out_of_range
    }

    Matrix<T> matrix(row_size, _cols);
    memcpy(matrix.data(), &_data[row_id * _cols], sizeof(T) * row_size * _cols);
    return matrix;
}

template<class T>
void Matrix<T>::set_row(const Matrix<T>& mat){
    set_row(0, mat);
}

template<class T>
void Matrix<T>::set_row(const std::size_t row_id, Matrix<T> mat){
    if(_cols != mat.cols()){
        //todo diff cols
    }
    if(row_id + mat.rows() > _rows){
        //todo out_of_range
    }

    memcpy(&_data[row_id * _cols], mat.data(), sizeof(T) * mat.get_size());
}

template<class T>
void Matrix<T>::insert_row(const Matrix<T>& mat){
    insert_row(_rows, mat);
}
template<class T>
void Matrix<T>::insert_row(const std::size_t row_id, const Matrix<T>& mat){
    
    if(get_size() == 0){
        set_data(mat.data(), mat.rows(), mat.cols());

        return;
    }

    if(_cols != mat.cols()){
        //todo diff cols
    }

    T* data = new T[(_rows + mat.rows()) * _cols];

    if(row_id > 0){
        memcpy(data, _data, sizeof(T) * row_id * _cols);
    }
    memcpy(&data[row_id * _cols], mat.data(), sizeof(T) * mat.get_size());
    if(row_id < _rows){
        memcpy(&data[(row_id + mat.rows()) * _cols], &_data[row_id * _cols], sizeof(T) * (_rows - row_id) * _cols);
    }

    set_shallow_data(data, _rows + mat.rows(), _cols);
}

template<class T>
Matrix<T> Matrix<T>::get_col(const std::size_t col_id, const std::size_t col_size){
    if(col_id + col_size > _cols){
        //todo out_of_range
    }

    Matrix<T> mat(_rows, col_size);
    for(std::size_t i = 0; i != _rows; i++){
        memcpy(&mat.data()[i*col_size], &_data[i * col_size + col_id], sizeof(T) * col_size);
    }
    return mat;
}

template<class T>
void Matrix<T>::set_col(const Matrix<T>& mat){
    set_col(0, mat);
}

template<class T>
void Matrix<T>::set_col(const std::size_t col_id, const Matrix<T>& mat){
    if(mat.rows() != _rows){
        //todo diff rows
    }

    if(col_id + mat.cols() > _cols){
        //todo out_of_range
    }

    T* sub_data   = mat.data();
    std::size_t sub_cols =  mat.cols();

    for(std::size_t i = 0; i != _rows; i++){
        memcpy(&_data[i * _cols + col_id], &sub_data[i * sub_cols], sizeof(T) * sub_cols);
    }
}

template<class T>
void Matrix<T>::insert_col(const Matrix<T>& mat){
    set_col(_cols, mat);
}

template<class T>
void Matrix<T>::insert_col(const std::size_t col_id, const Matrix<T>& mat){
    if(_rows != mat.rows()){
        //todo diff rows
    }

    if(col_id > _cols){
        //todo out_of_range
    }

    std::size_t sub_cols = mat.cols();
    std::size_t new_cols = _cols + sub_cols;
    T* sub_data   = mat.data();

    T* data = new T[_rows * new_cols];
    for(std::size_t i = 0; i != _rows; i++){
        if(col_id > 0){
            memcpy(&data[i * new_cols], &_data[i * _cols], sizeof(T) * sub_cols);
        }
        memcpy(&data[i * new_cols + sub_cols], &sub_data[i * sub_cols], sizeof(T) * sub_cols);
        if(col_id != _cols){
            memcpy(&data[i * new_cols + col_id + sub_cols], &_data[i * _cols + col_id], sizeof(T) * (_cols - col_id));
        }
    }

    set_shallow_data(data, _rows, new_cols);
}

template<class T>
Matrix<T> Matrix<T>::clone() const{
    Matrix<T> mat(_data, _rows, _cols);
    return mat;
}

template<class T>
void Matrix<T>::reset(const T& value){
   reset(value, _rows, _cols); 
}

template<class T>
void Matrix<T>::reset(const T& value,
                      const std::size_t rows,
                      const std::size_t cols){
    if(get_size() != rows * cols){
        if(_data != nullptr){
            delete[] _data;
            _data = nullptr;
        }
        _data = new T[rows * cols];
        _rows = rows;
        _cols = cols;
    }

    std::size_t size = rows * cols;
    if(value == 0 || value == -1){
        memset(_data, value, sizeof(T) * size);
    }else{
        auto lamda = [](T* a, const T& b){ *a = b; };
        ParallelOperator po;
        po.parallel_mul2one<T>(_data, size, value, lamda);
    }
}


template<class T>
void Matrix<T>::display(const std::string& split){
    printf("[%ld*%ld][\n", _rows, _cols);
    for(std::size_t i = 0; i != _rows; i++){
        printf("row[%ld][", i);
        for(std::size_t j = 0; j != _cols; j++){
            printf("%s", std::to_string(_data[i * _cols + j]).c_str());
            if(j != this->_cols - 1){
                printf("%s", split.c_str());
            }
        }
        printf("]\n");
    }
    printf("]\n");
}


template class Matrix<int>;
template class Matrix<real>;

}//namespace algebra
}//namespace abcdl
