/**********************************************
* Author: Jun Jiang - jiangjun4@sina.com
* Created: 2017-07-25 15:19
* Last modified: 2017-07-27 16:53
* Filename: MatrixHelper.h
* Description: matrix set
**********************************************/
#pragma once

#include "algebra/Matrix.h"
#include <vector>

namespace abcdl{
namespace algebra{

template<class T>
class MatrixSet{
public:
    void push_back(abcdl::algebra::Matrix<T> mat){
        _mats.push_back(mat);
    }

    abcdl::algebra::Matrix<T> operator [] (size_t idx) const{
        return _mats[idx];
    }

    inline const size_t size() const{
        return _mats.size();
    }

    inline const size_t rows() const{
        return  _mats.size() == 0 ? 0 : _mats[0].rows();
    }

    inline const size_t cols() const{
        return _mats.size() == 0 ? 0 : _mats[0].cols();
    }
private:
    std::vector<abcdl::algebra::Matrix<T>> _mats;
}; //class MatrixSet

typedef MatrixSet<real> MatSet;

} // algebra
} //abcdl
