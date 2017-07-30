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
void MatrixHelper<T>::dot(Matrix<T>& mat,
						  const Matrix<T>& mat_a,
						  const Matrix<T>& mat_b){
}

template<class T>
void MatrixHelper<T>::outer(Matrix<T>& mat,
							const Matrix<T>& mat_a,
							const Matrix<T>& mat_b){
}

template<class T>
void MatrixHelper<T>::pow(Matrix<T>& mat,
						  const Matrix<T>& mat_a,
						  const T& exponent){
    auto lamda = [](T* a, const T& b, const T& c){*a = std::pow(b, c);};
    po.parallel_mul2one_copy<T>(mat_a.data(), mat_a.get_size(), exponent, mat.data(), lamda);
}

template<class T>
void MatrixHelper<T>::log(Matrix<T>& mat, const Matrix<T>& mat_a){
    auto lamda = [](T* a, const T& b){ *a = std::log(b);};
    po.parallel_mul2one_copy<T>(mat_a.data(), mat_a.get_size(), mat.data(), lamda);
}

template<class T>
void MatrixHelper<T>::exp(Matrix<T>& mat, const Matrix<T>& mat_a){
    auto lamda = [](T* a, const T& b){ *a = std::exp(std::min(b, (T)EXP_MAX));};
    //auto lamda = [](T* a, const T& b){ *a = utils::TypeDef::exp<T>(b);};
    po.parallel_mul2one_copy<T>(mat_a.data(), mat_a.get_size(), mat.data(), lamda);
}

template<class T>
void MatrixHelper<T>::sigmoid(Matrix<T>& mat, const Matrix<T>& mat_a){
    auto lamda = [](T* a, const T& b){ *a = 1 / (1 + std::exp(-(std::min((T)SIGMOID_MAX, std::max(b, (T)SIGMOID_MIN)))));};
    po.parallel_mul2one_copy<T>(mat_a.data(), mat_a.get_size(), mat.data(), lamda);
}

template<class T>
void MatrixHelper<T>::softmax(Matrix<T>& mat, const Matrix<T>& mat_a){
//    auto lamda = [](T* a){ *a = std::log(*a);};
//    po.parallel_mul2one<T>(_data, get_size(), lamda);
}

template<class T>
void MatrixHelper<T>::tanh(Matrix<T>& mat, const Matrix<T>& mat_a){
    auto lamda = [](T* a, const T& b){ *a = 2.0 /(1.0 + std::exp(std::min((T)EXP_MAX, -2 * b))) - 1.0;};
    po.parallel_mul2one_copy<T>(mat_a.data(), mat_a.get_size(), mat.data(), lamda);
}

template<class T>
void MatrixHelper<T>::relu(Matrix<T>& mat, const Matrix<T>& mat_a){
    auto lamda = [](T* a, const T& b){ if(b < 0) {*a = 0;} else { *a = b;}};
    po.parallel_mul2one_copy<T>(mat_a.data(), mat_a.get_size(), mat.data(), lamda);
}

template<class T>
void MatrixHelper<T>::zero_like(Matrix<T>& mat, const Matrix<T>& mat_a){
	mat.set_data(mat_a.data(), mat_a.rows(), mat_a.cols()); 
}

template<class T>
void transpose(Matrix<T>& mat, const Matrix<T>& mat_a){
	if(&mat != &mat_a){
		mat.set_data(mat_a.data(), mat_a.rows(), mat_a.cols());
	}

	std::size_t rows = mat.rows();
	std::size_t cols = mat.cols();
	if(mat.rows() == 1 || mat.cols() == 1){
		mat.reshape(cols, rows);
	}else{

	}
}

template class MatrixHelper<int>;
template class MatrixHelper<real>;

}//namespace algebra
}//namespace abcdl
