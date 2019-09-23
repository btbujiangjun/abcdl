/***********************************************
 * Author: Jun Jiang - jiangjun4@sina.com
 * Create: 2017-07-26 15:15
 * Last modified : 2017-07-26 15:15
 * Filename      : Matrix.cpp
 * Description   : 
 **********************************************/
#include "algebra/Matrix.h"
#include "algebra/MatrixHelper.h"
#include "utils/Log.h"
#include "limits.h"

using abcdl::algebra::Mat;

int main(int argc,char** argv){
    abcdl::utils::log::set_min_log_level(abcdl::utils::log::INFO);
    abcdl::utils::log::initialize_log(argc, argv);
    
	abcdl::algebra::MatrixHelper<real> helper;
    
	real d1[6] = {1,2,-3,4,5,6};
    Mat m1(d1, 2, 3);

	auto m = helper.dot(m1, m1);
	m.display("^");	

	m1.display();

    auto m3 = m1 * m1;
    m3.display();

    m1 += m1;
    m1.display();

    m1.sigmoid();
    m1.display();

    Mat m2;
    helper.sigmoid_derivative(m2, m1);
    m2.display();

    m1 *= -1;
    m1.display();
    auto m4 = m2 * m1;
    m4.display();
}
