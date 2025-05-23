/***********************************************
 * Author: Jun Jiang - jiangjun4@sina.com
 * Create: 2017-07-25 16:33
 * Last modified : 2017-07-25 16:33
 * Filename      : Matrix.h
 * Description   : Base Data Structure
 **********************************************/

#include "algebra/Matrix.h"
#include "algebra/MatrixHelper.h"
#include <random>
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
Matrix<T>::Matrix(const size_t rows, const size_t cols){
    _rows = rows;
    _cols = cols;
    _data = new T[_rows * _cols];
    memset(_data, 0, sizeof(T) * _rows * _cols);
}

template<class T>
Matrix<T>::Matrix(const T& value,
                  const size_t rows,
                  const size_t cols){
    _rows = rows;
    _cols = cols;
    _data = new T[_rows * _cols];
    if(value == 0 || value == static_cast<T>(-1)){
        memset(_data, value, sizeof(T) * _rows * _cols);
    }else{
        auto lambda = [](T* a, const T& b){ *a = b; };
        _po.parallel_mul2one(_data, get_size(), value, lambda);
    }
}

template<class T>
Matrix<T>::Matrix(const T* data,
                  const size_t rows,
                  const size_t cols){
    _rows = rows;
    _cols = cols;
    _data = new T[_rows * _cols];
    memcpy(_data, data, sizeof(T) * _rows * _cols);
}

template<class T>
Matrix<T>::~Matrix(){
    if(_data != nullptr){
        delete[] _data;
        _data = nullptr;
    }
}

template<class T>
void Matrix<T>::set_shallow_data(T* data,
                                 const size_t rows,
                                 const size_t cols){
    if(_data != nullptr){
        delete[] _data;
    }
    _data = data;
    _rows = rows;
    _cols = cols;
}

template<class T>
Matrix<T> Matrix<T>::get_row(const size_t row_id, const size_t row_size) const{
    Matrix<T> mat;
	get_row(&mat, row_id, row_size);
	return mat;
}

template<class T>
void Matrix<T>::get_row(Matrix<T>* mat,
                        const size_t row_id,
                        const size_t row_size) const{
    CHECK(row_id + row_size <= _rows);
	T* data = new T[row_size * _cols];
	memcpy(data, &_data[row_id * _cols], sizeof(T) * row_size * _cols);
	mat->set_shallow_data(data, row_size, _cols);
}

template<class T>
void Matrix<T>::set_row(const Matrix<T>& mat){
    set_row(0, mat);
}

template<class T>
void Matrix<T>::set_row(const size_t row_id, const Matrix<T>& mat){
    CHECK(_cols == mat.cols());
    CHECK(row_id + mat.rows() <= _rows);
    memcpy(&_data[row_id * _cols], mat.data(), sizeof(T) * mat.get_size());
}

template<class T>
void Matrix<T>::insert_row(const Matrix<T>& mat){
    insert_row(_rows, mat);
}
template<class T>
void Matrix<T>::insert_row(const size_t row_id, const Matrix<T>& mat){  
    if(get_size() == 0){
        set_data(mat.data(), mat.rows(), mat.cols());
        return;
    }

    CHECK(_cols == mat.cols());

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
void Matrix<T>::swap_row(const size_t row_id1, const size_t row_id2){
    CHECK(row_id1 < _rows && row_id2 < _rows && row_id1 >=1 && row_id2 >= 1);
    if(row_id1 == row_id2){
        return;
    }
    T* data = new T[_cols];
    memcpy(&data, &_data[row_id1 * _cols], sizeof(T) * _cols);
    memcpy(&_data[row_id1 * _cols], &_data[row_id2 * _cols], sizeof(T) * _cols);
    memcpy(&_data[row_id2 * _cols], &data, sizeof(T) * _cols);
}

template<class T>
Matrix<T> Matrix<T>::get_col(const size_t col_id, const size_t col_size) const{
    Matrix<T> mat;
	get_col(&mat, col_id, col_size);
    return mat;
}

template<class T>
void Matrix<T>::get_col(Matrix<T>* mat,
                        const size_t col_id,
                        const size_t col_size) const{
    CHECK(col_id + col_size <= _cols);
	T* data = new T[_rows * col_size];
	for(size_t i = 0; i != _rows; i++){
		memcpy(&data[i * col_size], &_data[i * col_size + col_id], sizeof(T) * col_size);
	}
	mat->set_shallow_data(data, _rows, col_size);
}

template<class T>
void Matrix<T>::set_col(const Matrix<T>& mat){
    set_col(0, mat);
}

template<class T>
void Matrix<T>::set_col(const size_t col_id, const Matrix<T>& mat){
    CHECK(mat.rows() == _rows);
    CHECK(col_id + mat.cols() <= _cols);
    T* sub_data   = mat.data();
    size_t sub_cols =  mat.cols();

    for(size_t i = 0; i != _rows; i++){
        memcpy(&_data[i * _cols + col_id], &sub_data[i * sub_cols], sizeof(T) * sub_cols);
    }
}

template<class T>
void Matrix<T>::insert_col(const Matrix<T>& mat){
    set_col(_cols, mat);
}

template<class T>
void Matrix<T>::insert_col(const size_t col_id, const Matrix<T>& mat){
    
    if(get_size() == 0){
        set_data(mat.data(), mat.rows(), mat.cols());
        return;
    }

    CHECK(mat.rows() == _rows);
    CHECK(col_id <= _cols);

    size_t sub_cols = mat.cols();
    size_t new_cols = _cols + sub_cols;
    T* sub_data   = mat.data();

    T* data = new T[_rows * new_cols];
    for(size_t i = 0; i != _rows; i++){
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
void Matrix<T>::extend(const Matrix<T>& mat, const Axis_type type){
    if(type == Axis_type::COL){
        insert_col(mat);
    }else{
        insert_row(mat);
    }
}

template<class T>
Matrix<T> Matrix<T>::clone() const{
    return Matrix<T>(_data, _rows, _cols);
}
template<class T>
void Matrix<T>::clone(Matrix<T>& mat) const{
    mat.set_data(_data, _rows, _cols);
}

template<class T>
void Matrix<T>::reset(const T& value){
   reset(value, _rows, _cols); 
}

template<class T>
void Matrix<T>::reset(const T& value,
                      const size_t rows,
                      const size_t cols){
    size_t size = rows * cols;
    if(get_size() != size){
        if(_data != nullptr){
            delete[] _data;
        }
        _data = new T[size];
        _rows = rows;
        _cols = cols;
    }

    if(value == 0 || value == static_cast<T>(-1)){
        memset(_data, value, sizeof(T) * size);
    }else{
        auto lambda = [](T* a, const T& b){ *a = b; };
        _po.parallel_mul2one(_data, size, value, lambda);
    }
}

template<class T>
Matrix<T> Matrix<T>::Ts(){
    MatrixHelper<T> mh;
    Matrix<T> mat;
    mh.transpose(mat, *this);
    return mat;
}

template<class T>
Matrix<T>& Matrix<T>::transpose(){
    MatrixHelper<T> mh;
    mh.transpose(*this, *this);
	return *this;
}

template<class T>
void Matrix<T>::for_each(const std::function<void(T*)> &func){ 
    _po.parallel_mul2one(_data, get_size(), func);
}

template<class T>
void Matrix<T>::display(const std::string& split, const bool with_title) const{
    if(with_title) printf("[%ld*%ld][\n", _rows, _cols);
    for(size_t i = 0; i != _rows; i++){
        if(with_title) printf("row[%ld][", i);
        for(size_t j = 0; j != _cols; j++){
            printf("%s", std::to_string(_data[i * _cols + j]).c_str());
            if(j != this->_cols - 1){
                printf("%s", split.c_str());
            }
        }
        if(with_title){
            printf("]\n");
        }else{
            printf("\n");
        }
    }
    if(with_title) printf("]\n");
}


template<class T>
RandomMatrix<T>::RandomMatrix(size_t rows,
                              size_t cols,
                              const T& mean_value,
                              const T& stddev,
                              const T& min,
                              const T& max) : Matrix<T>(rows, cols){
    _mean_value = mean_value;
    _stddev     = stddev;
    _min        = min;
    _max        = max;

    reset();
}

template<class T>
void RandomMatrix<T>::reset(){
    T scale = _max - _min;
    T min   = _min;
    T max   = _max;

    size_t size = this->_rows * this->_cols;
    T* data = this->_data;
    size_t block_size = this->_po.get_block_size(size);
    size_t num_thread = this->_po.get_num_thread(size, block_size);
    std::vector<std::thread> threads(num_thread);
    std::default_random_engine engine(std::chrono::system_clock::now().time_since_epoch().count());
    std::normal_distribution<T> distribution(_mean_value, _stddev);

    for(size_t i = 0; i != num_thread; i++){
        threads[i] = std::thread(
            [&data, max, min, scale, &distribution, &engine](size_t start_idx, size_t end_idx){
                for(size_t ti = start_idx; ti != end_idx; ti++){
                    T value = static_cast<T>(distribution(engine));
                    if(max == min || value == max || value == min){
                        data[ti] = value;
                    }else if(value > max){
                        real step = (value - min)/scale;
                        value = min + (step - (int)step) * scale;
                    }else{
                        real step = (max - value)/scale;
                        value = min + (step - (int)step) * scale;
                    }
                }
            }, i * block_size, std::min(size, (i + 1) * block_size)
        );
    }

    for(auto& thread : threads){
        thread.join();
    }
}

template<class T>
void RandomMatrix<T>::reset(size_t rows,
                            size_t cols,
                            const T& mean_value,
                            const T& stddev,
                            const T& min,
                            const T& max){
    if(rows * cols != this->_rows * this->_cols){
        if(this->_data != nullptr){
            delete[] this->_data;
        }
        this->_data = new T[rows * cols];
    }

    this->_rows = rows;
    this->_cols = cols;
    _mean_value = mean_value;
    _stddev     = stddev;
    _min        = min;
    _max        = max;

    reset();
}


template class Matrix<int>;
template class Matrix<float>;
template class Matrix<double>;
template class Matrix<size_t>;

template class RandomMatrix<float>;
template class RandomMatrix<double>;

template class EyeMatrix<float>;
template class EyeMatrix<double>;
}//namespace algebra
}//namespace abcdl
