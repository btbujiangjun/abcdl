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
    abcdl::algebra::Matrix<real> m1(1.0, 10, 10);
    auto m2 = m1 + 2.0f;
    m2.log();
    m2.tanh();
    auto m3 = m2.get_row(2,2);
    m3 += m3;
    m3.display("|");
    m2.display("$");
    m1.display("@");
    m3 = m3.get_col(2, 2);
    m3 += m2.get_row(2, 2).get_col(2, 2);
    m3.display();
    m3 -= m3;
    m3.display();
}
