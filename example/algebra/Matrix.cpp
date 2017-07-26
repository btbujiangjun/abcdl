/***********************************************
 * Author: Jun Jiang - jiangjun4@sina.com
 * Create: 2017-07-26 15:15
 * Last modified : 2017-07-26 15:15
 * Filename      : Matrix.cpp
 * Description   : 
 **********************************************/
#include "algebra/Matrix.h"

using abcdl::algebra::Matrix;

int main(int argc,char** argv){
    abcdl::algebra::Matrix<real> m1(1.0, 5000, 5000);
    auto m2 = m1 + 2.0f;
    m2 += m2;
    m2 += m2;
    m2 += m2;
    m2 += m2;
    m2 += m1;
    m2 += 1;
//    m1.display();
    auto m3 = m1 + m2;
//    m1.display();
//    m2.display();
//    m3.display();
    m1 = 2.0f;
//    m1.display();
//    m2.display();
}
