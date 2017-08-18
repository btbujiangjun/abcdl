/***********************************************
 * Author: Jun Jiang - jiangjun4@sina.com
 * Create: 2017-07-26 15:15
 * Last modified : 2017-07-26 15:15
 * Filename      : Matrix.cpp
 * Description   : 
 **********************************************/
#include "algebra/Matrix.h"
#include "utils/Log.h"
#include "limits.h"

using abcdl::algebra::Matrix;

int main(int argc,char** argv){
    abcdl::utils::log::set_min_log_level(abcdl::utils::log::INFO);
    abcdl::utils::log::initialize_log(argc, argv);
    LOG(INFO) << "test INFO";
    LOG(WARNING) << "test WARNING";
    LOG(ERROR) << "test ERROR";
    LOG(FATAL) << "test FATAL";

    abcdl::algebra::Mat* mat1 = new abcdl::algebra::Mat(2, 3);
    abcdl::algebra::Mat* mat2 = new abcdl::algebra::Mat(2, 3);

    auto mat3 = *mat1 + *mat2;
    mat3.display();

    abcdl::algebra::RandomMatrix<real> m0(1000, 30000, 0, 1, -0.5, 0.5);
//    m0.display();
    abcdl::algebra::Mat m1;
    abcdl::algebra::Mat m2;
    real data1[6] = {1,2,3,4,5,6};
    real data2[6] = {1,2,3,4,5,6};
    m1.set_data(data1, 2, 3);
    m1.display("TT");
    m1.Ts().display();
    m2.set_data(data2, 3, 2);
    m1.get_data(20);
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
    m1.softmax();
    m1.display("|");
    m1.sigmoid();
    m1.display("|");
}
