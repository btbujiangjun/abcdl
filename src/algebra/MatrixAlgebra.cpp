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

template class Matrix<int>;
template class Matrix<real>;

}//namespace algebra
}//namespace abcdl
