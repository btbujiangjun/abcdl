/*********************************************
* Author: Jun Jiang - jiangjun4@sina.com
* Created: 2016-11-22 11:23
* Last modified: 2016-11-22 11:23
* Filename: TypeDef.h
* Description:
**********************************************/
#pragma once

#include <cmath>
#include <cstddef>
#include <typeinfo>
#include <stdlib.h>
#include <limits>

namespace abcdl{
namespace utils{
#ifdef ABCDL_TYPE_DOUBLE
    typedef double real;
    #define MAX_REAL std::numeric_limits<double>::max()
    #define MIN_REAL std::numeric_limits<double>::min()
    #define SOFTMAX_MIN -128.0
#else
    typedef float real;
    #define MAX_REAL std::numeric_limits<float>::max()
    #define MIN_REAL std::numeric_limits<float>::min()
    #define SOFTMAX_MIN -64.0
#endif

#define EXP_MAX 40.0
#define SIGMOID_MIN -13.0
#define SIGMOID_MAX 40.0
#define EPS 0.000001

//typedef size_t uint;
#define MAX_INT std::numeric_limits<int>::max();
#define MIN_INT std::numeric_limits<int>::min();

template<class T>
T type_cast(const char* data){
    if(typeid(T) == typeid(int)){
        return atoi(data);
    }else if(typeid(T) == typeid(real)){
        return atof(data);
    }else{
        return static_cast<T>(*data);
    }
}

template<class T1, class T2>
bool ccma_type_compare(){
    return sizeof(T1) > sizeof(T2);
}

template<class T>
T get_max_value(){
    if(typeid(T) == typeid(float)){
        return (T)std::numeric_limits<float>::max();
    }else if(typeid(T) == typeid(double)){
        return (T)std::numeric_limits<double>::max();
    }else{
        return (T)std::numeric_limits<int>::max();
    }
}
template<class T>
T get_min_value(){
    if(typeid(T) == typeid(float)){
        return (T)std::numeric_limits<float>::min();
    }else if(typeid(T) == typeid(double)){
        return (T)std::numeric_limits<double>::min();
    }else{
        return (T)std::numeric_limits<int>::min();
    }
}

template<class T>
static inline T exp(const T& value){
	return value > EXP_MAX ? std::exp(EXP_MAX) : std::exp(value);
}

template<class T>
static inline T sigmoid(const T& value){
	return static_cast<T>(1) / (1 + exp<T>(-value));
}


}//namespace utils
}//namespace abcdl

using abcdl::utils::real;
