/*********************************************
* Author: Jun Jiang - jiangjun4@sina.com
* Created: 2017-08-24 15:03
* Last modified: 2017-08-31 11:01
* Filename: Layer.h
* Description: Network Layer
**********************************************/
#pragma once

namespace abcdl{
namespace framework{

enum Layer_type{
	INPUT = 0,
    FULL_CONN,
	SUBSAMPLING,
	CONVOLUTION,
	OUTPUT
};

}//namespace framework
}//namespace abcdl
