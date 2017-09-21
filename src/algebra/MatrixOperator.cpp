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
	if(get_size() != mat.get_size()){
		return false;
	}

    bool result_value = true;
    auto lambda = [](bool* a, const T& b, const T& c){*a = (b == c);};
    utils::ParallelOperator po;
    po.parallel_reduce_boolean<T>(&result_value, _data, get_size(), mat.data(), mat.get_size(), lambda);
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
        if(get_size() != mat.get_size()){
            if(_data != nullptr){
                delete[] _data;
            }
        	_data = new T[mat.get_size()];
        }

        _rows = mat.rows();
        _cols = mat.cols();
        memcpy(_data, mat.data(), sizeof(T) * mat.get_size());
    }
    return *this;
}

template<class T>
Matrix<T> Matrix<T>::operator + (const T& value) const{
    auto lambda = [](T* a, const T& b){ if(b != 0){*a += b;} };
    auto mat = clone();
    ParallelOperator po;
    po.parallel_mul2one<T>(mat.data(), mat.get_size(), value, lambda);
    return mat;
}

template<class T>
Matrix<T> Matrix<T>::operator + (const Matrix<T>& mat_a) const{
    Matrix<T> mat;
    if(get_size() == 0){
        mat.set_data(mat_a);
        return mat;
    }

    CHECK(equal_shape(mat_a) || (_cols == mat_a.cols() && mat_a.rows() == 1));
    auto lambda = [](T* a, const T& b){ if(b != 0){*a += b;} };
    mat = clone();
    ParallelOperator po;
    po.parallel_mul2mul_repeat<T>(mat.data(), mat.get_size(), mat_a.data(), mat_a.get_size(), lambda);
    return mat;
}

template<class T>
Matrix<T>& Matrix<T>::operator += (const T& value){
    auto lambda = [](T* a, const T& b){ if(b != 0){ *a += b;} };
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

    CHECK(equal_shape(mat) || (_cols == mat.cols() && mat.rows() == 1));
    auto lambda = [](T* a, const T& b){ if(b != 0){*a += b;} };
    ParallelOperator po;
    po.parallel_mul2mul_repeat<T>(_data, get_size(), mat.data(), mat.get_size(), lambda);
    return *this;
}

template<class T>
Matrix<T> Matrix<T>::operator - (const T& value) const{
    auto lambda = [](T* a, const T& b){ if(b != 0){*a -= b;} };
    auto mat = clone();
    ParallelOperator po;
    po.parallel_mul2one<T>(mat.data(), mat.get_size(), value, lambda);
    return mat;
}

template<class T>
Matrix<T> Matrix<T>::operator - (const Matrix<T>& mat_a) const{
    CHECK(equal_shape(mat_a) || (_cols == mat_a.cols() && mat_a.rows() == 1));
    auto lambda = [](T* a, const T& b){ if(b != 0){*a -= b;} };
    auto mat = clone();
    ParallelOperator po;
    po.parallel_mul2mul_repeat<T>(mat.data(), get_size(), mat_a.data(), mat_a.get_size(), lambda);
    return mat;
}

template<class T>
Matrix<T>& Matrix<T>::operator -= (const T& value){
    auto lambda = [](T* a, const T& b){ if(b != 0){*a -= b;} };
    ParallelOperator po;
    po.parallel_mul2one<T>(_data, get_size(), value, lambda);    
	return *this;
}

template<class T>
Matrix<T>& Matrix<T>::operator -= (const Matrix<T>& mat){
    CHECK(equal_shape(mat) || (_cols == mat.cols() && mat.rows() == 1));
    auto lambda = [](T* a, const T& b){ if(b != 0){*a -= b;} };
    ParallelOperator po;
    po.parallel_mul2mul_repeat<T>(_data, get_size(), mat.data(), mat.get_size(), lambda);
	return *this;
}

template<class T>
Matrix<T> Matrix<T>::operator * (const T& value) const{
    auto lambda = [](T* a, const T& b){ if(b != 1){*a *= b;} };
    auto mat = clone();
    ParallelOperator po;
    po.parallel_mul2one<T>(mat.data(), get_size(), value, lambda);
    return mat;
}

template<class T>
Matrix<T> Matrix<T>::operator * (const Matrix<T>& mat_a) const{
    CHECK(equal_shape(mat_a) || (_cols == mat_a.cols() && mat_a.rows() == 1));
    auto lambda = [](T* a, const T& b){ if(b != 1){*a *= b;} };
    auto mat = clone();
    ParallelOperator po;
    po.parallel_mul2mul_repeat<T>(mat.data(), get_size(), mat_a.data(), mat_a.get_size(), lambda);
    return mat;
}

template<class T>
Matrix<T>& Matrix<T>::operator *= (const T& value){
    auto lambda = [](T* a, const T& b){ if(b != 1){*a *= b;} };
    ParallelOperator po;
    po.parallel_mul2one<T>(_data, get_size(), value, lambda);
	return *this;
}

template<class T>
Matrix<T>& Matrix<T>::operator *= (const Matrix<T>& mat){
    CHECK(equal_shape(mat) || (_cols == mat.cols() && mat.rows() == 1));
    auto lambda = [](T* a, const T& b){ if(b != 1){*a *= b;} };
    ParallelOperator po;
    po.parallel_mul2mul_repeat<T>(_data, get_size(), mat.data(), mat.get_size(), lambda);
	return *this;
}

template<class T>
Matrix<T> Matrix<T>::operator / (const T& value) const{
    auto lambda = [](T* a, const T& b){ if(b != 1){*a /= b;} };
    auto mat = clone();
    ParallelOperator po;
    po.parallel_mul2one<T>(mat.data(), get_size(), value, lambda);
    return mat;
}

template<class T>
Matrix<T> Matrix<T>::operator / (const Matrix<T>& mat_a) const{
    CHECK(equal_shape(mat_a) || (_cols == mat_a.cols() && mat_a.rows() == 1));
    auto lambda = [](T* a, const T& b){ if(b != 1){ *a /= b;} };
    auto mat = clone();
    ParallelOperator po;
    po.parallel_mul2mul_repeat<T>(mat.data(), mat.get_size(), mat_a.data(), mat_a.get_size(), lambda);
    return mat;
}

template<class T>
Matrix<T>& Matrix<T>::operator /= (const T& value){
    auto lambda = [](T* a, const T& b){ if(b != 1){*a /= b;} };
    ParallelOperator po;
    po.parallel_mul2one<T>(_data, get_size(), value, lambda);
	return *this;
}

template<class T>
Matrix<T>& Matrix<T>::operator /= (const Matrix<T>& mat){
    CHECK(equal_shape(mat) || (_cols == mat.cols() && mat.rows() == 1));
    auto lambda = [](T* a, const T& b){ if(b != 1){*a /= b;} };
    ParallelOperator po;
    po.parallel_mul2mul_repeat<T>(_data, get_size(), mat.data(), mat.get_size(), lambda);
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
}//namespace algebra
}//namespace abcdl
