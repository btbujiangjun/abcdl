/***********************************************
 * Author: Jun Jiang - jiangjun4@sina.com
 * Create: 2017-07-26 15:15
 * Last modified : 2017-07-26 15:15
 * Filename      : Matrix.cpp
 * Description   : 
 **********************************************/
#include "algebra/Matrix.h"
#include "limits.h"
#include "float.h"

using abcdl::algebra::Matrix;

int main(int argc,char** argv){

    /*
    abcdl::algebra::Matrix<double> m1(1.0, 10000, 10000);
	m1.set_data(0.2, 20);
	m1.set_data(2.3, 23);
	m1.set_data(13.4, 27000000);
	m1.set_data(14.4, 87000000);
	printf("sum is[%f] mean[%f]max[%f]min[%f]\n", (real)m1.sum(),(real) m1.mean(), (real)m1.max(), (real)m1.min());
	printf("sum is[%f]\n", (real)m1.sum());
*/

    abcdl::algebra::Matrix<real> m1;
    abcdl::algebra::Matrix<real> m2;
    real data1[6] = {1,2,3,4,5,6};
    real data2[6] = {1,2,3,4,5,6};
    m1.set_data(data1, 2, 3);
    m2.set_data(data2, 3, 2);
    m1.display("|");
    m2.display("|");
    m1.outer(m2);
    m1.log();
    m1.set_data(-10, 0);
    m1.set_data(-3, 2);
    m1.relu();
    m1.display("|");
    m1.transpose();
    m1.display("|");
    m1.set_data(data1, 2, 3);
    m1.display("|");
    m1.transpose();
    m1.display("|");


//    printf("Max of float is[%f], min is[%f]\n", FLT_MAX, FLT_MIN);

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
