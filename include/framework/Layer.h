/***********************************************
 * Author: Jun Jiang - jiangjun4@sina.com
 * Create: 2017-08-03 17:48
 * Last modified : 2017-08-03 17:48
 * Filename      : Layer.h
 * Description   : 
 **********************************************/
#pragma once

namespace abcdl{
namespace framework{

class Layer{
public:
    virtual void feedforward();
    virtual void backprapagation();
};//class Layer

}//namespace framework
}//namespace abcdl
