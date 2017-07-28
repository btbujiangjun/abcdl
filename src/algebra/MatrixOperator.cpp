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
Matrix<T>& Matrix<T>::operator = (const T& value){
    reset(value);
    return *this;
}

template<class T>
Matrix<T>& Matrix<T>::operator = (const Matrix<T>& mat){
    if(_data != nullptr){
        delete[] _data;
        _data = nullptr;
    }

    _rows = mat.rows();
    _cols = mat.cols();
    _data = new T[mat.get_size()];
    memcpy(_data, mat.data(), sizeof(T) * mat.get_size());
    return *this;
}

template<class T>
Matrix<T> Matrix<T>::operator + (const T& value){
    auto lamda = [](T* a, const T& b){ *a += b; };
    auto matrix = clone();
    ParallelOperator po;
    po.parallel_mul2one<T>(matrix.data(), matrix.get_size(), value, lamda);
    return matrix;
}

template<class T>
Matrix<T> Matrix<T>::operator + (const Matrix<T>& mat){
    if(!equal_shape(mat)){
        //todo diff size
    }

    auto lamda = [](T* a, const T& b){ *a += b; };
    auto matrix = clone();
    ParallelOperator po;
    po.parallel_mul2mul<T>(matrix.data(), matrix.get_size(), mat.data(), lamda);
    return matrix;
}

template<class T>
Matrix<T>& Matrix<T>::operator += (const T& value){
    auto lamda = [](T* a, const T& b){ *a += b; };
    ParallelOperator po;
    po.parallel_mul2one<T>(_data, get_size(), value, lamda);
    return *this;
}
template<class T>
Matrix<T>& Matrix<T>::operator += (const Matrix<T>& mat){
    if(!equal_shape(mat)){
        //todo diff size
    }
    auto lamda = [](T* a, const T& b){ *a += b; };
    ParallelOperator po;
    po.parallel_mul2mul<T>(_data, get_size(), mat.data(), lamda);
    return *this;
}

template<class T>
Matrix<T> Matrix<T>::operator - (const T& value){
    auto lamda = [](T* a, const T& b){ *a -= b; };
    auto matrix = clone();
    ParallelOperator po;
    po.parallel_mul2one<T>(matrix.data(), matrix.get_size(), value, lamda);
    return matrix;
}

template<class T>
Matrix<T> Matrix<T>::operator - (const Matrix<T>& mat){
    if(!equal_shape(mat)){
        //todo diff size
    }

    auto lamda = [](T* a, const T& b){ *a -= b; };
    auto matrix = clone();
    ParallelOperator po;
    po.parallel_mul2mul<T>(matrix.data(), get_size(), mat.data(), lamda);
    return matrix;
}

template<class T>
void Matrix<T>::operator -= (const T& value){
    auto lamda = [](T* a, const T& b){ *a -= b; };
    ParallelOperator po;
    po.parallel_mul2one<T>(_data, get_size(), value, lamda);    
}

template<class T>
void Matrix<T>::operator -= (const Matrix<T>& mat){
    if(!equal_shape(mat)){
        //todo diff size
    }

    auto lamda = [](T* a, const T& b){ *a -= b; };
    ParallelOperator po;
    po.parallel_mul2mul<T>(_data, get_size(), mat.data(), lamda);
}

template<class T>
Matrix<T> Matrix<T>::operator * (const T& value){
    auto lamda = [](T* a, const T& b){ *a *= b; };
    auto matrix = clone();
    ParallelOperator po;
    po.parallel_mul2one<T>(matrix.data(), get_size(), value, lamda);
    return matrix;
}

template<class T>
Matrix<T> Matrix<T>::operator * (const Matrix<T>& mat){
    if(!equal_shape(mat)){
        //todo diff size
    }

    auto lamda = [](T* a, const T& b){ *a *= b; };
    auto matrix = clone();
    ParallelOperator po;
    po.parallel_mul2mul<T>(matrix.data(), get_size(), mat.data(), lamda);
    return matrix;
}

template<class T>
void Matrix<T>::operator *= (const T& value){
    auto lamda = [](T* a, const T& b){ *a *= b; };
    ParallelOperator po;
    po.parallel_mul2one<T>(_data, get_size(), value, lamda);
}

template<class T>
void Matrix<T>::operator *= (const Matrix<T>& mat){
    if(!equal_shape(mat)){
        //todo diff size
    }

    auto lamda = [](T* a, const T& b){ *a *= b; };
    ParallelOperator po;
    po.parallel_mul2mul<T>(_data, get_size(), mat.data(), lamda);
}

template<class T>
Matrix<T> Matrix<T>::operator / (const T& value){
    auto lamda = [](T* a, const T& b){ *a /= b; };
    auto matrix = clone();
    ParallelOperator po;
    po.parallel_mul2one<T>(matrix.data(), get_size(), value, lamda);
    return matrix;
}

template<class T>
Matrix<T> Matrix<T>::operator / (const Matrix<T>& mat){
    if(!equal_shape(mat)){
        //todo diff size
    }

    auto matrix = clone();
    auto lamda = [](T* a, const T& b){ *a /= b; };
    ParallelOperator po;
    po.parallel_mul2mul<T>(matrix.data(), mat.get_size(), mat.data(), lamda);
    return matrix;
}

template<class T>
void Matrix<T>::operator /= (const T& value){
    auto lamda = [](T* a, const T& b){ *a /= b; };
    ParallelOperator po;
    po.parallel_mul2one<T>(_data, get_size(), value, lamda);
}

template<class T>
void Matrix<T>::operator /= (const Matrix<T>& mat){
    if(!equal_shape(mat)){
        //todo diff size
    }

    auto lamda = [](T* a, const T& b){ *a /= b; };
    ParallelOperator po;
    po.parallel_mul2mul<T>(_data, get_size(), mat.data(), lamda);
}

template class Matrix<int>;
template class Matrix<real>;
}//namespace algebra
}//namespace abcdl
