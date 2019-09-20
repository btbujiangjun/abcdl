/***********************************************
 * Author: Jun Jiang - jiangjun4@sina.com
 * Create: 2017-07-26 15:15
 * Last modified : 2017-07-26 15:15
 * Filename      : Matrix.cpp
 * Description   : 
 **********************************************/
#include "algebra/Matrix.h"
#include "algebra/MatrixHelper.h"
#include "utils/LibsvmHelper.h"

using abcdl::algebra::Mat;

int main(int argc,char** argv){
    abcdl::utils::log::set_min_log_level(abcdl::utils::log::INFO);
    abcdl::utils::log::initialize_log(argc, argv);
    abcdl::algebra::Mat data_mat;
    abcdl::algebra::Mat label_mat;
    abcdl::utils::LibsvmHelper<real> helper;
    helper.read_data(251, 2, "./data/test.libsvm", &data_mat, &label_mat);
    printf("sample size:%zu\n", data_mat.rows());
    helper.write_data("./data/test.write.libsvm", data_mat, label_mat);
}
