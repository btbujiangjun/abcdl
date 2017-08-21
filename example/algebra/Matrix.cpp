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
    real d1[6] = {1,2,3,4,5,6};
    real d2[6] = {1,2,3,4,5,6};
    Mat m1(d1, 2, 3);
    Mat m2(d2, 3, 2);

    m1.display();
    m2.display();

    abcdl::algebra::MatrixHelper<real> helper;
    auto m3 = helper.dot(m1, m2);
    m3.display();
}
