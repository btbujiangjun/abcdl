/***********************************************
 * Author: Jun Jiang - jiangjun4@sina.com
 * Create: 2017-07-26 13:27
 * Last modified : 2017-07-26 13:27
 * Filename      : MatrixOperator.cpp
 * Description   : 
 **********************************************/

#include "algebra/Matrix.h"
#include "utils/ParallelOperator.h"
#include <string.h>


using abcdl::utils::ParallelOperator;

namespace abcdl{
namespace algebra{

template<class T>
void Matrix<T>::operator= (const T& value){
    reset(value);
}

template<class T>
Matrix<T> Matrix<T>::operator+ (const T& value) const{
    uint size = get_size();
    T* data = new T[size];
    memcpy(data, _data, sizeof(T) * size);
    
    auto lamda = [](T* a, const T& b){ *a += b; };
    ParallelOperator po;
    po.parallel_mul2one<T>(data, size, value, lamda);
    
    Matrix<T> mat;
    mat.set_shallow_data(data, _rows, _cols);

    return mat;
}

template<class T>
Matrix<T> Matrix<T>::operator+ (Matrix<T>& mat) const{
    if(_rows != mat.rows() || _cols != mat.cols()){
        //todo diff size
    }

    uint size = get_size();
    T* data = new T[size];
    memcpy(data, _data, sizeof(T) * size);
    
    T* sub_data = mat.data();

    auto lamda = [](T* a, const T& b){ *a += b; };
    ParallelOperator po;
    po.parallel_mul2mul<T>(data, size, sub_data, lamda);

    Matrix<T> matrix;
    matrix.set_shallow_data(data, _rows, _cols);
    return matrix;
}

template<class T>
void Matrix<T>::operator+= (const T& value){
    auto lamda = [](T* a, const T& b){ *a += b; };
    ParallelOperator po;
    po.parallel_mul2one<T>(_data, get_size(), value, lamda);
    
}
template<class T>
void Matrix<T>::operator+= (Matrix<T>& mat){
    if(_rows != mat.rows() || _cols != mat.cols()){
        //todo diff size
    }

    uint size = get_size();
    T* sub_data = mat.data();

    auto lamda = [](T* a, const T& b){ *a += b; };
    ParallelOperator po;
    po.parallel_mul2mul<T>(_data, size, sub_data, lamda);

}
template class Matrix<int>;
template class Matrix<real>;
}//namespace algebra
}//namespace abcdl
