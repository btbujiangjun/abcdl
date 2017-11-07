/*********************************************
* Author: Jun Jiang - jiangjun4@sina.com
* Created: 2017-09-28 15:22
* Last modified: 2017-09-28 15:22
* Filename: Loss.h
* Description: loss function
**********************************************/
#pragma once

#include "algebra/Matrix.h"

namespace abcdl{
namespace framework{

class Loss{
public:
    virtual ~Loss() = default;
    virtual real loss(const abcdl::algebra::Mat& label,
                      const abcdl::algebra::Mat& activate) = 0;
};//class Loss

class MSELoss : public Loss{
public:
    real loss(const abcdl::algebra::Mat& label,
              const abcdl::algebra::Mat& activate) override{
        auto diff_mat = label - activate;
        return (diff_mat * diff_mat).sum()/2;
    }
};//class MSELoss

}//namespace framework
}//namespace abcdl
