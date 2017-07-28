/***********************************************
 * Author: Jun Jiang - jiangjun4@sina.com
 * Create: 2017-07-27 14:48
 * Last modified : 2017-07-27 17:14
 * Filename      : MatrixHelper.cpp
 * Description   : 
 **********************************************/

#include "algebra/MatrixHelper.h"
#include "utils/ParallelOperator.h"
#include <cmath>
#include <string.h>

namespace abcdl{
namespace algebra{

template<class T>
Matrix<T> MatrixHelper<T>::dot(const Matrix<T>& mat_a, const Matrix<T>& mat_b){
    auto matrix = zero_like(mat_a);
    dot(mat_a, mat_b, matrix);
    return matrix;
}
template<class T>
void MatrixHelper<T>::dot(const Matrix<T>& mat_a, const Matrix<T>& mat_b, Matrix<T>& mat){
}

template<class T>
Matrix<T> MatrixHelper<T>::outer(const Matrix<T>& mat_a, const Matrix<T>& mat_b){
    auto matrix = zero_like(mat_a);
    outer(mat_a, mat_b, matrix);
    return matrix;
}
template<class T>
void MatrixHelper<T>::outer(const Matrix<T>& mat_a, const Matrix<T>& mat_b, Matrix<T>& mat){
}

template<class T>
Matrix<T> MatrixHelper<T>::pow(const Matrix<T>& mat_a, const T& exponent){
    auto matrix = zero_like(mat_a);
    pow(mat_a, exponent, matrix);
    return matrix;
}
template<class T>
void MatrixHelper<T>::pow(const Matrix<T>& mat_a, const T& exponent, Matrix<T>& mat){
    auto lamda = [](T* a, const T& b, const T& c){*a = std::pow(b, c);};
    po.parallel_mul2one_copy<T>(mat_a.data(), mat_a.get_size(), exponent, mat.data(), lamda);
}

template<class T>
Matrix<T> MatrixHelper<T>::log(const Matrix<T>& mat_a){
    auto matrix = zero_like(mat_a);
    log(mat_a, matrix);
    return matrix;
}
template<class T>
void MatrixHelper<T>::log(const Matrix<T>& mat_a, Matrix<T>& mat){
    auto lamda = [](T* a, const T& b){ *a = std::log(b);};
    po.parallel_mul2one_copy<T>(mat_a.data(), mat_a.get_size(), mat.data(), lamda);
}

template<class T>
Matrix<T> MatrixHelper<T>::exp(const Matrix<T>& mat_a){
    auto matrix = zero_like(mat_a);
    exp(mat_a, matrix);
    return matrix;
}
template<class T>
void MatrixHelper<T>::exp(const Matrix<T>& mat_a, Matrix<T>& mat){
    auto lamda = [](T* a, const T& b){ *a = std::exp(std::min(b, (T)EXP_MAX));};
    po.parallel_mul2one_copy<T>(mat_a.data(), mat_a.get_size(), mat.data(), lamda);
}

template<class T>
Matrix<T> MatrixHelper<T>::sigmoid(const Matrix<T>& mat_a){
    auto matrix = zero_like(mat_a);
    sigmoid(mat_a, matrix);
    return matrix;
}
template<class T>
void MatrixHelper<T>::sigmoid(const Matrix<T>& mat_a, Matrix<T>& mat){
    auto lamda = [](T* a, const T& b){ *a = 1 / (1 + std::exp(-(std::min((T)SIGMOID_MAX, std::max(b, (T)SIGMOID_MIN)))));};
    po.parallel_mul2one_copy<T>(mat_a.data(), mat_a.get_size(), mat.data(), lamda);
}

template<class T>
Matrix<T> MatrixHelper<T>::softmax(const Matrix<T>& mat_a){
    auto matrix = zero_like(mat_a);
    softmax(mat_a, matrix);
    return matrix;
}
template<class T>
void MatrixHelper<T>::softmax(const Matrix<T>& mat_a, Matrix<T>& mat){
//    auto lamda = [](T* a){ *a = std::log(*a);};
//    po.parallel_mul2one<T>(_data, get_size(), lamda);
}

template<class T>
Matrix<T> MatrixHelper<T>::tanh(const Matrix<T>& mat_a){
    auto matrix = zero_like(mat_a);
    tanh(mat_a, matrix);
    return matrix;
}
template<class T>
void MatrixHelper<T>::tanh(const Matrix<T>& mat_a, Matrix<T>& mat){
    auto lamda = [](T* a, const T& b){ *a = 2.0 /(1.0 + std::exp(std::min((T)EXP_MAX, -2 * b))) - 1.0;};
    po.parallel_mul2one_copy<T>(mat_a.data(), mat_a.get_size(), mat.data(), lamda);
}

template<class T>
Matrix<T> MatrixHelper<T>::relu(const Matrix<T>& mat_a){
    auto matrix = mat_a.clone();
    relu(mat_a, matrix);
    return matrix;
}
template<class T>
void MatrixHelper<T>::relu(const Matrix<T>& mat_a, Matrix<T>& mat){
    auto lamda = [](T* a, const T& b){ if(b < 0) *a = 0;};
    po.parallel_mul2one_copy<T>(mat_a.data(), mat_a.get_size(), mat.data(), lamda);
}

template<class T>
Matrix<T> MatrixHelper<T>::zero_like(const Matrix<T>& mat){
    Matrix<T> matrix(mat.rows(), mat.cols());
    return matrix;
}


template class MatrixHelper<int>;
template class MatrixHelper<real>;

}//namespace algebra
}//namespace abcdl
