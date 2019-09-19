/***********************************************
 * Author: Jun Jiang - jiangjun4@sina.com
 * Create: 2017-07-26 13:27
 * Last modified : 2017-07-26 13:27
 * Filename      : MatrixOperator.cpp
 * Description   : 
 **********************************************/

#include <string.h>
#include "algebra/Matrix.h"
#include "utils/ParallelOperator.h"


using abcdl::utils::ParallelOperator;

namespace abcdl{
namespace algebra{

template<class T>
T& Matrix<T>::operator [] (const size_t idx) const{
    CHECK(idx < get_size());
	return _data[idx];
}

template<class T>
bool Matrix<T>::operator == (const Matrix<T>& mat) const{
	if(this == &mat){
		return true;
	}
    size_t size = mat.get_size();
	if(get_size() != size){
		return false;
	}

    bool result_value = true;
    auto lambda = [](bool* a, const T& b, const T& c){*a = (*a && (b == c));};
    utils::ParallelOperator po;
    po.parallel_reduce_boolean<T>(&result_value, _data, get_size(), mat.data(), size, lambda);
	return result_value;
}

template<class T>
Matrix<T>& Matrix<T>::operator = (const T& value){
    reset(value);
    return *this;
}

template<class T>
Matrix<T>& Matrix<T>::operator = (const Matrix<T>& mat){
    if(this != &mat){
        size_t size = mat.get_size();
        if(get_size() != size){
            if(_data != nullptr){
                delete[] _data;
            }
        	_data = new T[size];
        }

        _rows = mat.rows();
        _cols = mat.cols();
        memcpy(_data, mat.data(), sizeof(T) * size);
    }
    return *this;
}

template<class T>
Matrix<T> Matrix<T>::operator + (const T& value) const{
    auto new_mat = clone();
    if (value != 0){
        auto lambda = [](T* a, const T& b){*a += b;};
        ParallelOperator po;
        po.parallel_mul2one<T>(new_mat.data(), new_mat.get_size(), value, lambda);
    }
    return new_mat;
}

template<class T>
Matrix<T> Matrix<T>::operator + (const Matrix<T>& mat) const{
    if(get_size() == 0){
        return mat.clone();
    }

    CHECK(equal_shape(mat) 
          || (_cols == mat.cols() && mat.rows() == 1)
          || (_rows == mat.rows() && mat.cols() == 1));
    Matrix<T> new_mat = clone();

    if(equal_shape(mat) || (_cols == mat.cols() && mat.rows() == 1)){
        auto lambda = [](T* a, const T& b){*a += b;};
        ParallelOperator po;
        po.parallel_mul2mul_repeat<T>(new_mat.data(), new_mat.get_size(), mat.data(), mat.get_size(), lambda);
    }else{
        for(size_t i = 0; i < _rows; i++){
            T value = mat.get_data(i, 0);
            for(size_t j = 0; j < _cols; j++){
                new_mat.set_data(get_data(i, j) + value, i, j);
            }
        }
    }
    return new_mat;
}

template<class T>
Matrix<T>& Matrix<T>::operator += (const T& value){
    auto lambda = [](T* a, const T& b){*a += b;};
    ParallelOperator po;
    po.parallel_mul2one<T>(_data, get_size(), value, lambda);
    return *this;
}
template<class T>
Matrix<T>& Matrix<T>::operator += (const Matrix<T>& mat){
    if(get_size() == 0){
        set_data(mat);
        return *this;
    }

    CHECK(equal_shape(mat) 
          || (_cols == mat.cols() && mat.rows() == 1
          || (_rows == mat.rows() && mat.cols() == 1)));
    
    if(equal_shape(mat) || (_cols == mat.cols() && mat.rows() == 1)){
        auto lambda = [](T* a, const T& b){*a += b;};
        ParallelOperator po;
        po.parallel_mul2mul_repeat<T>(_data, get_size(), mat.data(), mat.get_size(), lambda);
    }else{
        for(size_t i = 0; i < _rows; i++){
            T value = mat.get_data(i, 0);
            for(size_t j = 0; j < _cols; j++){
                set_data(get_data(i, j) + value, i, j);
            }
        }
    }
    return *this;
}

template<class T>
Matrix<T> Matrix<T>::operator - (const T& value) const{
    auto lambda = [](T* a, const T& b){*a -= b;};
    auto new_mat = clone();
    ParallelOperator po;
    po.parallel_mul2one<T>(new_mat.data(), new_mat.get_size(), value, lambda);
    return new_mat;
}

template<class T>
Matrix<T> Matrix<T>::operator - (const Matrix<T>& mat) const{
    CHECK(equal_shape(mat) 
          || (_cols == mat.cols() && mat.rows() == 1
          || (_rows == mat.rows() && mat.cols() == 1)));
    
    auto new_mat = clone();
    if(equal_shape(mat) || (_cols == mat.cols() && mat.rows() == 1)){
        auto lambda = [](T* a, const T& b){*a -= b;};
        ParallelOperator po;
        po.parallel_mul2mul_repeat<T>(new_mat.data(), new_mat.get_size(), mat.data(), mat.get_size(), lambda);
    }else{
        for(size_t i = 0; i < _rows; i++){
            T value = mat.get_data(i, 0);
            for(size_t j = 0; j < _cols; j++){
                new_mat.set_data(get_data(i, j) - value, i, j);
            }
        }
    }
    return new_mat;
}

template<class T>
Matrix<T>& Matrix<T>::operator -= (const T& value){
    auto lambda = [](T* a, const T& b){*a -= b;};
    ParallelOperator po;
    po.parallel_mul2one<T>(_data, get_size(), value, lambda);    
	return *this;
}

template<class T>
Matrix<T>& Matrix<T>::operator -= (const Matrix<T>& mat){
    CHECK(equal_shape(mat) 
          || (_cols == mat.cols() && mat.rows() == 1
          || (_rows == mat.rows() && mat.cols() == 1)));
    
    if(equal_shape(mat) || (_cols == mat.cols() && mat.rows() == 1)){
        auto lambda = [](T* a, const T& b){*a -= b;};
        ParallelOperator po;
        po.parallel_mul2mul_repeat<T>(_data, get_size(), mat.data(), mat.get_size(), lambda);
    }else{
        for(size_t i = 0; i < _rows; i++){
            T value = mat.get_data(i, 0);
            for(size_t j = 0; j < _cols; j++){
                set_data(get_data(i, j) - value, i, j);
            }
        }
    }
    return *this;
}

template<class T>
Matrix<T> Matrix<T>::operator * (const T& value) const{
    auto lambda = [](T* a, const T& b){*a *= b;};
    auto new_mat = clone();
    ParallelOperator po;
    po.parallel_mul2one<T>(new_mat.data(), new_mat.get_size(), value, lambda);
    return new_mat;
}

template<class T>
Matrix<T> Matrix<T>::operator * (const Matrix<T>& mat) const{
    CHECK(equal_shape(mat) 
          || (_cols == mat.cols() && mat.rows() == 1
          || (_rows == mat.rows() && mat.cols() == 1)));
    
    auto new_mat = clone();
    if(equal_shape(mat) || (_cols == mat.cols() && mat.rows() == 1)){
        auto lambda = [](T* a, const T& b){*a *= b;};
        ParallelOperator po;
        po.parallel_mul2mul_repeat<T>(new_mat.data(), new_mat.get_size(), mat.data(), mat.get_size(), lambda);
    }else{
        for(size_t i = 0; i < _rows; i++){
            T value = mat.get_data(i, 0);
            for(size_t j = 0; j < _cols; j++){
                new_mat.set_data(get_data(i, j) * value, i, j);
            }
        }
    }
    return new_mat;
}

template<class T>
Matrix<T>& Matrix<T>::operator *= (const T& value){
    auto lambda = [](T* a, const T& b){*a *= b;};
    ParallelOperator po;
    po.parallel_mul2one<T>(_data, get_size(), value, lambda);
	return *this;
}

template<class T>
Matrix<T>& Matrix<T>::operator *= (const Matrix<T>& mat){
    CHECK(equal_shape(mat) 
          || (_cols == mat.cols() && mat.rows() == 1
          || (_rows == mat.rows() && mat.cols() == 1)));
    
    if(equal_shape(mat) || (_cols == mat.cols() && mat.rows() == 1)){
        auto lambda = [](T* a, const T& b){*a *= b;};
        ParallelOperator po;
        po.parallel_mul2mul_repeat<T>(_data, get_size(), mat.data(), mat.get_size(), lambda);
    }else{
        for(size_t i = 0; i < _rows; i++){
            T value = mat.get_data(i, 0);
            for(size_t j = 0; j < _cols; j++){
                set_data(get_data(i, j) * value, i, j);
            }
        }
    }
    return *this;
}

template<class T>
Matrix<T> Matrix<T>::operator / (const T& value) const{
    auto lambda = [](T* a, const T& b){ if(b != 0){*a /= b;} };
    auto new_mat = clone();
    ParallelOperator po;
    po.parallel_mul2one<T>(new_mat.data(), get_size(), value, lambda);
    return new_mat;
}

template<class T>
Matrix<T> Matrix<T>::operator / (const Matrix<T>& mat) const{
    CHECK(equal_shape(mat) 
          || (_cols == mat.cols() && mat.rows() == 1
          || (_rows == mat.rows() && mat.cols() == 1)));
    
    auto new_mat = clone();
    if(equal_shape(mat) || (_cols == mat.cols() && mat.rows() == 1)){
        auto lambda = [](T* a, const T& b){ CHECK(b != 0); *a /= b;};
        ParallelOperator po;
        po.parallel_mul2mul_repeat<T>(new_mat.data(), new_mat.get_size(), mat.data(), mat.get_size(), lambda);
    }else{
        for(size_t i = 0; i < _rows; i++){
            T value = mat.get_data(i, 0);
            CHECK(value != 0);
            for(size_t j = 0; j < _cols; j++){
                new_mat.set_data(get_data(i, j) / value, i, j);
            }
        }
    }
    return new_mat;
}

template<class T>
Matrix<T>& Matrix<T>::operator /= (const T& value){
    auto lambda = [](T* a, const T& b){*a /= b;};
    ParallelOperator po;
    po.parallel_mul2one<T>(_data, get_size(), value, lambda);
	return *this;
}

template<class T>
Matrix<T>& Matrix<T>::operator /= (const Matrix<T>& mat){
    CHECK(equal_shape(mat) 
          || (_cols == mat.cols() && mat.rows() == 1
          || (_rows == mat.rows() && mat.cols() == 1)));
    
    if(equal_shape(mat) || (_cols == mat.cols() && mat.rows() == 1)){
        auto lambda = [](T* a, const T& b){*a /= b;};
        ParallelOperator po;
        po.parallel_mul2mul_repeat<T>(_data, get_size(), mat.data(), mat.get_size(), lambda);
    }else{
        for(size_t i = 0; i < _rows; i++){
            T value = mat.get_data(i, 0);
            CHECK(value != 0);
            for(size_t j = 0; j < _cols; j++){
                set_data(get_data(i, j) / value, i, j);
            }
        }
    }
    return *this;
}

template<class T>
Matrix<T>::operator Matrix<int>() const{
    auto lambda = [](int* a, const T& b){*a = static_cast<int>(b);};
    Matrix<int> mat(_rows, _cols);
    ParallelOperator po;
    po.parallel_mul2mul<int, T>(mat.data(), mat.get_size(), _data, get_size(), lambda);
    return mat;
}

template<class T>
Matrix<T>::operator Matrix<float>() const{
    auto lambda = [](float* a, const T& b){*a = static_cast<float>(b);};
    Matrix<float> mat(_rows, _cols);
    ParallelOperator po;
    po.parallel_mul2mul<float, T>(mat.data(), mat.get_size(), _data, get_size(), lambda);
    return mat;
}

template<class T>
Matrix<T>::operator Matrix<double>() const{
    auto lambda = [](double* a, const T& b){*a = static_cast<double>(b);};
    Matrix<double> mat(_rows, _cols);
    ParallelOperator po;
    po.parallel_mul2mul<double, T>(mat.data(), mat.get_size(), _data, get_size(), lambda);
    return mat;
}

template<class T>
Matrix<T>::operator Matrix<size_t>() const{
    auto lambda = [](size_t* a, const T& b){*a = static_cast<size_t>(b);};
    Matrix<size_t> mat(_rows, _cols);
    ParallelOperator po;
    po.parallel_mul2mul<size_t, T>(mat.data(), mat.get_size(), _data, get_size(), lambda);
    return mat;
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
