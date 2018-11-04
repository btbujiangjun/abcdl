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
        _matset.push_back(mat);
    }

    abcdl::algebra::Matrix<T> operator [] (size_t idx) const{
        return _matset[idx];
    }

    inline const size_t size() const{
        return _matset.size();
    }

    inline const size_t rows() const{
        if( _matset.size() == 0 ){
            return 0;
        }
        return _matset[0].rows();
    }

    inline const size_t cols() const{
        if( _matset.size() == 0){
            return 0;
        }
        return _matset[0].cols();
    }
private:
    std::vector<abcdl::algebra::Matrix<T>> _matset;
}; //class MatrixSet

typedef MatrixSet<real> MatSet;

} // algebra
} //abcdl
