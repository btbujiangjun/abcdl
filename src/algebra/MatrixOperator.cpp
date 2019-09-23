/***********************************************
 * Author: Jun Jiang - jiangjun4@sina.com
 * Create: 2017-07-26 13:27
 * Last modified : 2017-07-26 13:27
 * Filename      : MatrixOperator.cpp
 * Description   : 
 **********************************************/

#include <string.h>
#include "algebra/Matrix.h"

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
    auto lambda = [](bool* a, const T& b, const T& c){*a = (b == c);};
    _po.parallel_reduce_boolean(&result_value, _data, get_size(), mat.data(), size, lambda);
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
        _po.parallel_mul2one(new_mat.data(), new_mat.get_size(), value, lambda);
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
        _po.parallel_mul2mul_repeat(new_mat.data(), new_mat.get_size(), mat.data(), mat.get_size(), lambda);
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
    _po.parallel_mul2one(_data, get_size(), value, lambda);
    return *this;
}
template<class T>
Matrix<T>& Matrix<T>::operator += (const Matrix<T>& mat){
    if(get_size() == 0){
        set_data(mat);
        return *this;
    }

    CHECK(equal_shape(mat) 
          || (_cols == mat.cols() && mat.rows() == 1)
          || (_rows == mat.rows() && mat.cols() == 1));
    
    if(equal_shape(mat) || (_cols == mat.cols() && mat.rows() == 1)){
        auto lambda = [](T* a, const T& b){*a += b;};
        _po.parallel_mul2mul_repeat(_data, get_size(), mat.data(), mat.get_size(), lambda);
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
    _po.parallel_mul2one(new_mat.data(), new_mat.get_size(), value, lambda);
    return new_mat;
}

template<class T>
Matrix<T> Matrix<T>::operator - (const Matrix<T>& mat) const{
    CHECK(equal_shape(mat) 
          || (_cols == mat.cols() && mat.rows() == 1)
          || (_rows == mat.rows() && mat.cols() == 1));
    
    auto new_mat = clone();
    if(equal_shape(mat) || (_cols == mat.cols() && mat.rows() == 1)){
        auto lambda = [](T* a, const T& b){*a -= b;};
        _po.parallel_mul2mul_repeat(new_mat.data(), new_mat.get_size(), mat.data(), mat.get_size(), lambda);
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
    _po.parallel_mul2one(_data, get_size(), value, lambda);    
	return *this;
}

template<class T>
Matrix<T>& Matrix<T>::operator -= (const Matrix<T>& mat){
    CHECK(equal_shape(mat) 
          || (_cols == mat.cols() && mat.rows() == 1)
          || (_rows == mat.rows() && mat.cols() == 1));
    
    if(equal_shape(mat) || (_cols == mat.cols() && mat.rows() == 1)){
        auto lambda = [](T* a, const T& b){*a -= b;};
        _po.parallel_mul2mul_repeat(_data, get_size(), mat.data(), mat.get_size(), lambda);
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
    _po.parallel_mul2one(new_mat.data(), new_mat.get_size(), value, lambda);
    return new_mat;
}

template<class T>
Matrix<T> Matrix<T>::operator * (const Matrix<T>& mat) const{
    CHECK(equal_shape(mat) 
          || (_cols == mat.cols() && mat.rows() == 1)
          || (_rows == mat.rows() && mat.cols() == 1));
    
    auto new_mat = clone();
    if(equal_shape(mat) || (_cols == mat.cols() && mat.rows() == 1)){
        auto lambda = [](T* a, const T& b){*a *= b;};
        _po.parallel_mul2mul_repeat(new_mat.data(), new_mat.get_size(), mat.data(), mat.get_size(), lambda);
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
    _po.parallel_mul2one(_data, get_size(), value, lambda);
	return *this;
}

template<class T>
Matrix<T>& Matrix<T>::operator *= (const Matrix<T>& mat){
    CHECK(equal_shape(mat) 
          || (_cols == mat.cols() && mat.rows() == 1)
          || (_rows == mat.rows() && mat.cols() == 1));
    
    if(equal_shape(mat) || (_cols == mat.cols() && mat.rows() == 1)){
        auto lambda = [](T* a, const T& b){*a *= b;};
        _po.parallel_mul2mul_repeat(_data, get_size(), mat.data(), mat.get_size(), lambda);
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
	CHECK(value != 0);
    auto lambda = [](T* a, const T& b){*a /= b;};
    auto new_mat = clone();
    _po.parallel_mul2one(new_mat.data(), new_mat.get_size(), value, lambda);
    return new_mat;
}

template<class T>
Matrix<T> Matrix<T>::operator / (const Matrix<T>& mat) const{
    CHECK(equal_shape(mat) 
          || (_cols == mat.cols() && mat.rows() == 1)
          || (_rows == mat.rows() && mat.cols() == 1));
    
    auto new_mat = clone();
    if(equal_shape(mat) || (_cols == mat.cols() && mat.rows() == 1)){
        auto lambda = [](T* a, const T& b){ CHECK(b != 0); *a /= b;};
        _po.parallel_mul2mul_repeat(new_mat.data(), new_mat.get_size(), mat.data(), mat.get_size(), lambda);
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
	CHECK(value != 0);
    auto lambda = [](T* a, const T& b){*a /= b;};
    _po.parallel_mul2one(_data, get_size(), value, lambda);
	return *this;
}

template<class T>
Matrix<T>& Matrix<T>::operator /= (const Matrix<T>& mat){
    CHECK(equal_shape(mat) 
          || (_cols == mat.cols() && mat.rows() == 1)
          || (_rows == mat.rows() && mat.cols() == 1));
    
    if(equal_shape(mat) || (_cols == mat.cols() && mat.rows() == 1)){
        auto lambda = [](T* a, const T& b){ CHECK(b != 0); *a /= b;};
        _po.parallel_mul2mul_repeat(_data, get_size(), mat.data(), mat.get_size(), lambda);
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
    Matrix<int> mat(_rows, _cols);
	size_t size = get_size();
	for(size_t i = 0; i < size; i++){
		mat.set_data(i, static_cast<int>(get_data(i)));
	}
    return mat;
}

template<class T>
Matrix<T>::operator Matrix<float>() const{
    Matrix<float> mat(_rows, _cols);
	size_t size = get_size();
	for(size_t i = 0; i < size; i++){
		mat.set_data(i, static_cast<float>(get_data(i)));
	}
    return mat;
}

template<class T>
Matrix<T>::operator Matrix<double>() const{
    Matrix<double> mat(_rows, _cols);
	size_t size = get_size();
	for(size_t i = 0; i < size; i++){
		mat.set_data(i, static_cast<double>(get_data(i)));
	}
    return mat;
}

template<class T>
Matrix<T>::operator Matrix<size_t>() const{
    Matrix<size_t> mat(_rows, _cols);
	size_t size = get_size();
	for(size_t i = 0; i < size; i++){
		mat.set_data(i, static_cast<size_t>(get_data(i)));
	}
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
