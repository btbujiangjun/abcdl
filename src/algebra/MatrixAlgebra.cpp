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
Matrix<T>& Matrix<T>::dot(const Matrix<T>& mat){
    MatrixHelper<T> mh;
    mh.dot(*this, *this, mat);
	return *this;	
}

template<class T>
Matrix<T>& Matrix<T>::outer(const Matrix<T>& mat){
    MatrixHelper<T> mh;
    mh.outer(*this, *this, mat);
	return *this;
}

template<class T>
Matrix<T>& Matrix<T>::pow(const T& exponent){
    MatrixHelper<T> mh;
    mh.pow(*this, *this, exponent);
	return *this;
}

template<class T> 
Matrix<T>& Matrix<T>::log(){
    MatrixHelper<T> mh;
    mh.log(*this, *this);
	return *this;
}

template<class T>
Matrix<T>& Matrix<T>::exp(){
    MatrixHelper<T> mh;
    mh.exp(*this, *this);
	return *this;
}

template<class T>
Matrix<T>& Matrix<T>::sigmoid(){
    MatrixHelper<T> mh;
    mh.sigmoid(*this, *this);
	return *this;
}

template<class T>
Matrix<T>& Matrix<T>::softmax(){
    MatrixHelper<T> mh;
    mh.softmax(*this, *this);
	return *this;
}

template<class T>
Matrix<T>& Matrix<T>::tanh(){
    MatrixHelper<T> mh;
    mh.tanh(*this, *this);
	return *this;
}

template<class T>
Matrix<T>& Matrix<T>::relu(){
    MatrixHelper<T> mh;
    mh.relu(*this, *this);
	return *this;
}

template<class T>
Matrix<T>& Matrix<T>::expand(const size_t row_dim, const size_t col_dim){
    MatrixHelper<T> mh;
    mh.expand(*this, *this, row_dim, col_dim);
	return *this;
}

template<class T>
bool Matrix<T>::convn(const Matrix<T>& kernal,
                      const size_t stride,
                      const Convn_type type){
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
Matrix<size_t> Matrix<T>::argmax(Axis_type axis_type) const{
    size_t size = (axis_type == Axis_type::ROW)? _rows : _cols;
   	size_t end_idx = (axis_type == Axis_type::ROW) ? _cols : _rows;
    size_t* idx_data = new size_t[size];
    for(size_t i = 0; i != size; i++){
        T max_value = 0;
        size_t max_idx = 0;
        for(size_t j = 0; j != end_idx; j++){
            T value = (axis_type == Axis_type::ROW) ? _data[i * _cols + j] : _data[j * _cols + i];
            if( j == 0 || value > max_value){
                max_value = value;
                max_idx = j;
            }
        }
        idx_data[i] = max_idx;
    }

    size_t rows = (axis_type == Axis_type::ROW) ? size : 1;
    size_t cols = (axis_type == Axis_type::ROW) ? 1 : size;

	abcdl::algebra::Matrix<size_t> mat;
	mat.set_shallow_data(idx_data, rows, cols);

    return mat;
}

template<class T>
size_t Matrix<T>::argmax(const size_t id, const Axis_type axis_type) const{
    bool is_row = (axis_type == abcdl::algebra::Axis_type::ROW);
	if(is_row){
        CHECK(id < _rows);
    }else{
        CHECK(id < _cols);
    }

    size_t max_idx 	= 0;
	T max 			= is_row ? get_data(id, 0) : get_data(0, id);
	size_t size 	= is_row ? _cols : _rows;
	
	for(size_t i = 0; i != size; i++){
		T value = is_row ? get_data(id, i) : get_data(i, id);
		if(value > max){
			max = value;
			max_idx = i;
		}
	}
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
template class Matrix<size_t>;

}//namespace algebra
}//namespace abcdl
