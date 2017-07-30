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
    abcdl::algebra::Matrix<real> m1(1.0, 10000, 10000);

	printf("sum is[%d]\n", (int)m1.sum());
	
	/*
	auto m2 = m1 + 2.0f;
    m2.log();
    m2.tanh();
	m2 = m2 - 1;
	m2 = m2 * 2;
	m2 = m2 / 0.7;
    m2 += m2;
    m2 += m2;
    m2 += m2;
    m2 += m2;
    m2 += m2;
    m2 += m2;
    m2 += m2;
    m2 += m2;
    m2 += m2;
    m2 += m2;
    m2 += m2;
    m2 += m2;
    m2 += m2;
    m2 += m2;
    m2 += m2;
    m2 += m2;
    m2 *= m2;
    m2 /= m2;
    m2 += m2;
    m2 += m2;

	Matrix<real> m3;
    m2.get_row(m3, 2, 2);
    m3 += m3;
    Matrix<real> m4;
	m3.get_col(m4, 2, 2);
    m4.display();
    m4 -= m4;
    m4.display();
	*/
}
