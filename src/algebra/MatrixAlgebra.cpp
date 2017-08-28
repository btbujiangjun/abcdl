/***********************************************
 * Author: Jun Jiang - jiangjun4@sina.com
 * Create: 2017-07-27 14:48
 * Last modified : 2017-07-27 14:48
 * Filename      : MatrixAlgebra.cpp
 * Description   : 
 **********************************************/

#include "algebra/MatrixHelper.h"

namespace abcdl{
namespace algebra{

template<class T>
void Matrix<T>::dot(const Matrix<T>& mat){
    MatrixHelper<T> mh;
    mh.dot(*this, *this, mat);    
}

template<class T>
void Matrix<T>::outer(const Matrix<T>& mat){
    MatrixHelper<T> mh;
    mh.outer(*this, *this, mat);
}

template<class T>
void Matrix<T>::pow(const T& exponent){
    MatrixHelper<T> mh;
    mh.pow(*this, *this, exponent);
}

template<class T> 
void Matrix<T>::log(){
    MatrixHelper<T> mh;
    mh.log(*this, *this);
}

template<class T>
void Matrix<T>::exp(){
    MatrixHelper<T> mh;
    mh.exp(*this, *this);
}

template<class T>
void Matrix<T>::sigmoid(){
    MatrixHelper<T> mh;
    mh.sigmoid(*this, *this);
}

template<class T>
void Matrix<T>::softmax(){
    MatrixHelper<T> mh;
    mh.softmax(*this, *this);
}

template<class T>
void Matrix<T>::tanh(){
    MatrixHelper<T> mh;
    mh.tanh(*this, *this);
}

template<class T>
void Matrix<T>::relu(){
    MatrixHelper<T> mh;
    mh.relu(*this, *this);
}

template<class T>
bool convn(const Matrix<T>& kernal,
           const size_t stride,
           const abcdl::algebra::Convn_type type){
    MatrixHelper<T> mh;
    return mh.convn(*this, *this, kernal, stride, type); 
}

template<class T>
T Matrix<T>::max() const{
	if(get_size() == 0){
		return 0;
	}
	T max = _data[0];
	auto lamda = [](T* a, const T& b){if(b > *a){*a = b;}};
	utils::ParallelOperator po;
	po.parallel_reduce_mul2one<T>(&max, _data, get_size(), lamda);
	return max;   
}

template<class T>
size_t Matrix<T>::argmax() const{
	if(get_size() == 0){
		return 0;
	}
	T max = _data[0];
    size_t max_idx = 0;
	auto lamda = [](T* a, const T& b, size_t* max_idx, const size_t idx){if(b > *a){*a = b; *max_idx = idx;}};
	utils::ParallelOperator po;
	po.parallel_reduce_mul2one<T>(&max, &max_idx, _data, get_size(), lamda);
	return max_idx;   
}

template<class T>
T Matrix<T>::min() const{
	if(get_size() == 0){
		return 0;
	}
	T min = _data[0];
	auto lamda = [](T* a, const T& b){if(b < *a){*a = b;}};
	utils::ParallelOperator po;
	po.parallel_reduce_mul2one<T>(&min, _data, get_size(), lamda);
	return min;   
}

template<class T>
size_t Matrix<T>::argmin() const{
	if(get_size() == 0){
		return 0;
	}
	T min = _data[0];
    size_t min_idx = 0;
	auto lamda = [](T* a, const T& b, size_t* min_idx, const size_t idx){if(b < *a){*a = b; *min_idx = idx;}};
	utils::ParallelOperator po;
	po.parallel_reduce_mul2one<T>(&min, &min_idx, _data, get_size(), lamda);
	return min_idx;   
}

template<class T>
T Matrix<T>::sum() const{
	T sum = 0;
	auto lamda = [](T* a, const T& b){if(b != 0){ *a += b;} };
	utils::ParallelOperator po;
	po.parallel_reduce_mul2one<T>(&sum, _data, get_size(), lamda);
	return sum;   
}

template<class T>
real Matrix<T>::mean() const{
	return ((real)sum())/get_size();
}

/*
template<class T>
bool Matrix<T>::inverse(Matrix<T>& mat){

}
*/

template class Matrix<int>;
template class Matrix<float>;
template class Matrix<double>;

}//namespace algebra
}//namespace abcdl
