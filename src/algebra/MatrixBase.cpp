/***********************************************
 * Author: Jun Jiang - jiangjun4@sina.com
 * Create: 2017-07-25 16:33
 * Last modified : 2017-07-25 16:33
 * Filename      : Matrix.h
 * Description   : Base Data Structure
 **********************************************/

#include "algebra/Matrix.h"
#include <string.h>
#include <iostream>
#include <stdio.h>

namespace abcdl{
namespace algebra{

template<class T>
Matrix<T>::Matrix(){
    _rows = 0;
    _cols = 0;
    _data = nullptr;
}

template<class T>
Matrix<T>::Matrix(const uint rows, const uint cols){
    _rows = rows;
    _cols = cols;
    _data = new T[_rows * _cols];
    memset(_data, 0, sizeof(T) * _rows * _cols);
}

template<class T>
Matrix<T>::Matrix(const T& value,
                  const uint rows,
                  const uint cols){
    _rows = rows;
    _cols = cols;
    _data = new T[_rows * _cols];
    if(value == 0 || value == -1){
        memset(_data, value, sizeof(T) * _rows * _cols);
    }else{
        uint size = get_size();
        for(uint i = 0; i != size; i++){
            _data[i] = value;
        }
    }
}

template<class T>
Matrix<T>::Matrix(const T* data,
                  const uint rows,
                  const uint cols){
    _rows = rows;
    _cols = cols;
    _data = new T[_rows * _cols];
    memcpy(_data, data, sizeof(T) * _rows * _cols);
}

template<class T>
Matrix<T>::~Matrix(){
    if(_data != nullptr){
        delete[] _data;
    }
}

template<class T>
T Matrix<T>::get_data(const uint idx){
    return _data[idx];
}

template<class T>
T Matrix<T>::get_data(const uint row_id, const uint col_id){
    return _data[row_id * _cols + col_id];
}

template<class T>
void Matrix<T>::set_data(const T& value, const uint idx){
    _data[idx] = value;
}

template<class T>
void Matrix<T>::set_data(const T& value,
                         const uint row_id,
                         const uint col_id){
    _data[row_id * _cols + col_id] = value; 
}

template<class T>
void Matrix<T>::set_data(const T* data,
                         const uint rows,
                         const uint cols){
    uint size = rows * cols;
    if(_rows * _cols != size){
        if(_data != nullptr){
            delete[] _data;
        }
        _data = new T[size];
    }

    memcpy(_data, data, sizeof(T) * size);
    _rows = rows;
    _cols = cols;
}

template<class T>
void Matrix<T>::set_data(Matrix<T>& mat){
    const T* data = mat.data();
    set_data(data, mat.rows(), mat.cols());
}

template<class T>
void Matrix<T>::set_shallow_data(T* data,
                                 const uint rows,
                                 const uint cols){
    if(_data != nullptr){
        delete[] _data;
    }
    _data = data;
    _rows = rows;
    _cols = cols;
}

template<class T>
Matrix<T> Matrix<T>::get_row(const uint row_id, const uint row_size){
    if(row_id + row_size > _rows){
        //todo out_of_range
    }

    T* data = new T[row_size * _cols];
    memcpy(data, &_data[row_id * _cols], sizeof(T) * row_size * _cols);

    Matrix<T> mat;
    mat.set_shallow_data(data, row_size, _cols);

    return mat;
}

template<class T>
void Matrix<T>::set_row(Matrix<T>& mat){
    set_row(0, mat);
}

template<class T>
void Matrix<T>::set_row(const uint row_id, Matrix<T> mat){
    if(_cols != mat.cols()){
        //todo diff cols
    }
    if(row_id + mat.rows() > _rows){
        //todo out_of_range
    }

    memcpy(&_data[row_id * _cols], mat.data(), sizeof(T) * mat.get_size());
}

template<class T>
void Matrix<T>::insert_row(Matrix<T>& mat){
    insert_row(_rows, mat);
}
template<class T>
void Matrix<T>::insert_row(const uint row_id, Matrix<T>& mat){
    
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
Matrix<T> Matrix<T>::get_col(const uint col_id, const uint col_size){
    if(col_id + col_size > _cols){
        //todo out_of_range
    }

    T* data = new T[_rows * col_size];
    for(uint i = 0; i != _rows; i++){
        memcpy(&data[i*col_size], &_data[i * col_size + col_id], sizeof(T) * col_size);
    }

    Matrix<T> mat;
    mat.set_shallow_data(data, _rows, col_size);
    return mat;
}

template<class T>
void Matrix<T>::set_col(Matrix<T>& mat){
    set_col(0, mat);
}

template<class T>
void Matrix<T>::set_col(const uint col_id, Matrix<T>& mat){
    if(mat.rows() != _rows){
        //todo diff rows
    }

    if(col_id + mat.cols() > _cols){
        //todo out_of_range
    }

    T* sub_data   = mat.data();
    uint sub_cols =  mat.cols();

    for(uint i = 0; i != _rows; i++){
        memcpy(&_data[i * _cols + col_id], &sub_data[i * sub_cols], sizeof(T) * sub_cols);
    }
}

template<class T>
void Matrix<T>::insert_col(Matrix<T>& mat){
    set_col(_cols, mat);
}

template<class T>
void Matrix<T>::insert_col(const uint col_id, Matrix<T>& mat){
    if(_rows != mat.rows()){
        //todo diff rows
    }

    if(col_id > _cols){
        //todo out_of_range
    }

    uint sub_cols = mat.cols();
    uint new_cols = _cols + sub_cols;
    T* sub_data   = mat.data();

    T* data = new T[_rows * new_cols];
    for(uint i = 0; i != _rows; i++){
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
Matrix<T> Matrix<T>::clone(){
    Matrix<T> mat(_data, _rows, _cols);
    return mat;
}

template<class T>
void Matrix<T>::reset(const T& value){
   reset(value, _rows, _cols); 
}

template<class T>
void Matrix<T>::reset(const T& value,
                      const uint rows,
                      const uint cols){
    if(get_size() != rows * cols){
        if(_data != nullptr){
            delete[] _data;
        }
        _data = new T[rows * cols];
        _rows = rows;
        _cols = cols;
    }

    uint size = rows * cols;
    if(value == 0 || value == -1){
        memset(_data, value, sizeof(T) * size);
    }else{
        for(uint i = 0; i != size; i++){
            _data[i] = value;
        }
    }
}


template<class T>
void Matrix<T>::display(const std::string& split){
    printf("[%d*%d][\n", _rows, _cols);
    for(uint i = 0; i != _rows; i++){
        printf("row[%d][", i);
        for(uint j = 0; j != _cols; j++){
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
