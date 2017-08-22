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
    auto lamda = [](bool* a, const T& b, const T& c){*a = (b == c);};
    utils::ParallelOperator po;
    po.parallel_reduce_boolean<T>(&result_value, _data, get_size(), mat.data(), mat.get_size(), lamda);
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
    auto lamda = [](T* a, const T& b){ if(b != 0){*a += b;} };
    Matrix<T> mat;
    clone(mat);
    ParallelOperator po;
    po.parallel_mul2one<T>(mat.data(), mat.get_size(), value, lamda);
    return mat;
}

template<class T>
Matrix<T> Matrix<T>::operator + (const Matrix<T>& mat_a) const{
    CHECK(equal_shape(mat_a));
    auto lamda = [](T* a, const T& b){ if(b != 0){*a += b;} };
    Matrix<T> mat;
    clone(mat);
    ParallelOperator po;
    po.parallel_mul2mul<T>(mat.data(), mat.get_size(), mat_a.data(), mat_a.get_size(), lamda);
    return mat;
}

template<class T>
Matrix<T>& Matrix<T>::operator += (const T& value){
    auto lamda = [](T* a, const T& b){ if(b != 0){ *a += b;} };
    ParallelOperator po;
    po.parallel_mul2one<T>(_data, get_size(), value, lamda);
    return *this;
}
template<class T>
Matrix<T>& Matrix<T>::operator += (const Matrix<T>& mat){
    if(get_size() == 0){
        set_data(mat);
        return *this;
    }
    CHECK(equal_shape(mat));
    auto lamda = [](T* a, const T& b){ if(b != 0){*a += b;} };
    ParallelOperator po;
    po.parallel_mul2mul<T>(_data, get_size(), mat.data(), mat.get_size(), lamda);
    return *this;
}

template<class T>
Matrix<T> Matrix<T>::operator - (const T& value) const{
    auto lamda = [](T* a, const T& b){ if(b != 0){*a -= b;} };
    Matrix<T> mat;
    clone(mat);
    ParallelOperator po;
    po.parallel_mul2one<T>(mat.data(), mat.get_size(), value, lamda);
    return mat;
}

template<class T>
Matrix<T> Matrix<T>::operator - (const Matrix<T>& mat_a) const{
    CHECK(equal_shape(mat_a));
    auto lamda = [](T* a, const T& b){ if(b != 0){*a -= b;} };
    Matrix<T> mat;
    clone(mat);
    ParallelOperator po;
    po.parallel_mul2mul<T>(mat.data(), get_size(), mat_a.data(), mat_a.get_size(), lamda);
    return mat;
}

template<class T>
Matrix<T>& Matrix<T>::operator -= (const T& value){
    auto lamda = [](T* a, const T& b){ if(b != 0){*a -= b;} };
    ParallelOperator po;
    po.parallel_mul2one<T>(_data, get_size(), value, lamda);    
	return *this;
}

template<class T>
Matrix<T>& Matrix<T>::operator -= (const Matrix<T>& mat){
    CHECK(equal_shape(mat));
    auto lamda = [](T* a, const T& b){ if(b != 0){*a -= b;} };
    ParallelOperator po;
    po.parallel_mul2mul<T>(_data, get_size(), mat.data(), mat.get_size(), lamda);
	return *this;
}

template<class T>
Matrix<T> Matrix<T>::operator * (const T& value) const{
    auto lamda = [](T* a, const T& b){ if(b != 0){*a *= b;} };
    Matrix<T> mat;
    clone(mat);
    ParallelOperator po;
    po.parallel_mul2one<T>(mat.data(), get_size(), value, lamda);
    return mat;
}

template<class T>
Matrix<T> Matrix<T>::operator * (const Matrix<T>& mat_a) const{
    CHECK(equal_shape(mat_a));
    auto lamda = [](T* a, const T& b){ *a *= b; };
    Matrix<T> mat;
    clone(mat);
    ParallelOperator po;
    po.parallel_mul2mul<T>(mat.data(), get_size(), mat.data(), mat.get_size(), lamda);
    return mat;
}

template<class T>
Matrix<T>& Matrix<T>::operator *= (const T& value){
    auto lamda = [](T* a, const T& b){ *a *= b; };
    ParallelOperator po;
    po.parallel_mul2one<T>(_data, get_size(), value, lamda);
	return *this;
}

template<class T>
Matrix<T>& Matrix<T>::operator *= (const Matrix<T>& mat){
    CHECK(equal_shape(mat));
    auto lamda = [](T* a, const T& b){ *a *= b; };
    ParallelOperator po;
    po.parallel_mul2mul<T>(_data, get_size(), mat.data(), mat.get_size(), lamda);
	return *this;
}

template<class T>
Matrix<T> Matrix<T>::operator / (const T& value) const{
    auto lamda = [](T* a, const T& b){ *a /= b; };
    Matrix<T> mat;
    clone(mat);
    ParallelOperator po;
    po.parallel_mul2one<T>(mat.data(), get_size(), value, lamda);
    return mat;
}

template<class T>
Matrix<T> Matrix<T>::operator / (const Matrix<T>& mat_a) const{
    CHECK(equal_shape(mat_a));
    auto lamda = [](T* a, const T& b){ *a /= b; };
    Matrix<T> mat;
    clone(mat);
    ParallelOperator po;
    po.parallel_mul2mul<T>(mat.data(), mat.get_size(), mat_a.data(), mat_a.get_size(), lamda);
    return mat;
}

template<class T>
Matrix<T>& Matrix<T>::operator /= (const T& value){
    auto lamda = [](T* a, const T& b){ *a /= b; };
    ParallelOperator po;
    po.parallel_mul2one<T>(_data, get_size(), value, lamda);
	return *this;
}

template<class T>
Matrix<T>& Matrix<T>::operator /= (const Matrix<T>& mat){
    CHECK(equal_shape(mat));
    auto lamda = [](T* a, const T& b){ *a /= b; };
    ParallelOperator po;
    po.parallel_mul2mul<T>(_data, get_size(), mat.data(), mat.get_size(), lamda);
	return *this;
}


template class Matrix<int>;
template class Matrix<float>;
template class Matrix<double>;
}//namespace algebra
}//namespace abcdl
