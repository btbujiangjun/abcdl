/***********************************************
 * Author: Jun Jiang - jiangjun4@sina.com
 * Create: 2017-07-27 14:48
 * Last modified : 2017-07-27 17:14
 * Filename      : MatrixHelper.cpp
 * Description   : 
 **********************************************/

#include "algebra/MatrixHelper.h"
#include "utils/ParallelOperator.h"
#include <cmath>
#include <string.h>

namespace abcdl{
namespace algebra{

template<class T>
void MatrixHelper<T>::dot(Matrix<T>& mat,
						  const Matrix<T>& mat_a,
						  const Matrix<T>& mat_b){
    std::size_t row_a = mat_a.rows();
    std::size_t col_a = mat_a.cols();
    std::size_t row_b = mat_b.rows();
    std::size_t col_b = mat_b.cols();

    if(col_a != row_b){
        //todo dim error
        printf("Dot Dim Error[%ld:%ld][%ld:%ld]\n", row_a,col_a, row_b, col_b);
    }

    T* data   = new T[row_a * col_b];
    T* data_a = mat_a.data();
    T* data_b = mat_b.data();

    std::size_t size = row_a * col_b * col_a;
    std::size_t num_thread = po.get_num_thread(size, po.get_block_size(size));
    std::size_t block_size = row_a / num_thread;
    if(row_a % num_thread != 0){
        block_size += 1;
    }

    std::vector<std::thread> threads(num_thread);
    for(std::size_t i = 0; i != num_thread; i++){
        threads[i] = std::thread(
            [&data, &data_a, &data_b, &col_a, &col_b](std::size_t start_idx, std::size_t end_idx){
				for(std::size_t ti = start_idx; ti < end_idx; ti++){
					std::size_t a_init_idx = ti * col_a;
					for(std::size_t tj = 0; tj != col_b; tj++){
						T value = 0;
						std::size_t a_idx = a_init_idx;
						for(std::size_t tk = 0; tk != col_a; tk++){
							value += data_a[a_idx++] * data_b[tk * col_b + tj];
						}
						data[ti * col_b + tj] = value;
					}
				}
            }, i * block_size , std::min(row_a, (i + 1) * block_size)
        );
    }

    for(auto& thread : threads){
        thread.join();
    }

    mat.set_shallow_data(data, row_a, col_b);
}

template<class T>
void MatrixHelper<T>::outer(Matrix<T>& mat,
							const Matrix<T>& mat_a,
							const Matrix<T>& mat_b){
    std::size_t size_a = mat_a.get_size();
    std::size_t size_b = mat_b.get_size();
    T* data_a = mat_a.data();
    T* data_b = mat_b.data();
    T* data   = new T[size_a * size_b];
    auto lamda = [](T* a, const T& b, const T& c){*a = b * c;};
    po.parallel_mul2mul_cross<T>(data, data_a, size_a, data_b, size_b, lamda);
    mat.set_shallow_data(data, size_a, size_b);
}

template<class T>
void MatrixHelper<T>::pow(Matrix<T>& mat,
						  const Matrix<T>& mat_a,
						  const T& exponent){
    auto lamda = [](T* a, const T& b, const T& c){*a = std::pow(b, c);};
    po.parallel_mul2one_copy<T>(mat.data(), mat_a.data(), mat_a.get_size(), exponent, lamda);
}

template<class T>
void MatrixHelper<T>::log(Matrix<T>& mat, const Matrix<T>& mat_a){
    auto lamda = [](T* a, const T& b){ *a = std::log(b);};
    po.parallel_mul2one_copy<T>(mat.data(), mat_a.data(), mat_a.get_size(), lamda);
}

template<class T>
void MatrixHelper<T>::exp(Matrix<T>& mat, const Matrix<T>& mat_a){
    auto lamda = [](T* a, const T& b){ *a = std::exp(std::min(b, (T)EXP_MAX));};
    po.parallel_mul2one_copy<T>(mat.data(), mat_a.data(), mat_a.get_size(), lamda);
}

template<class T>
void MatrixHelper<T>::sigmoid(Matrix<T>& mat, const Matrix<T>& mat_a){
    auto lamda = [](T* a, const T& b){ *a = 1 / (1 + std::exp(-(std::min((T)SIGMOID_MAX, std::max(b, (T)SIGMOID_MIN)))));};
    po.parallel_mul2one_copy<T>(mat.data(), mat_a.data(), mat_a.get_size(), lamda);
}

template<class T>
void MatrixHelper<T>::softmax(Matrix<T>& mat, const Matrix<T>& mat_a){
    T max = mat_a.max();
    if(&mat != &mat_a){
        mat.set_data(mat_a.data(), mat_a.rows(), mat_a.cols());
    }

    auto lamda = [](T* a, const T& max){*a = std::exp(std::max((*a - max), (T)SOFTMAX_MIN));};
    po.parallel_mul2one<T>(mat.data(), mat.get_size(), max, lamda);
    mat /=  mat.sum();
}

template<class T>
void MatrixHelper<T>::tanh(Matrix<T>& mat, const Matrix<T>& mat_a){
    auto lamda = [](T* a, const T& b){ *a = 2.0 /(1.0 + std::exp(std::min((T)EXP_MAX, -2 * b))) - 1.0;};
    po.parallel_mul2one_copy<T>(mat.data(), mat_a.data(), mat_a.get_size(), lamda);
}

template<class T>
void MatrixHelper<T>::relu(Matrix<T>& mat, const Matrix<T>& mat_a){
    auto lamda = [](T* a, const T& b){ if(b < 0) {*a = 0;} else { *a = b;}};
    if(&mat != &mat_a){
        mat.reset(0, mat_a.rows(), mat_a.cols());
    }
    po.parallel_mul2one_copy<T>(mat.data(), mat_a.data(), mat_a.get_size(), lamda);
}

template<class T>
void MatrixHelper<T>::zero_like(Matrix<T>& mat, const Matrix<T>& mat_a){
	mat.reset(0, mat_a.rows(), mat_a.cols()); 
}


template<class T>
void MatrixHelper<T>::transpose(Matrix<T>& mat, const Matrix<T>& mat_a){
	std::size_t rows = mat_a.rows();
	std::size_t cols = mat_a.cols();
	if((rows == 1 || cols == 1) &&(&mat != &mat_a)){
		mat.set_data(mat_a.data(), rows, cols);
		mat.reshape(cols, rows);
        return ;
	}

    T* src_data      = mat_a.data();
    T* data          = new T[mat_a.get_size()];
    std::size_t num_thread = po.get_num_thread(mat_a.get_size(), po.get_block_size(mat_a.get_size()));
    std::size_t block_size = cols / num_thread;
    if(cols % num_thread != 0){
        block_size += 1;
    }
        
    std::vector<std::thread> threads(num_thread);
    for(std::size_t i = 0; i != num_thread; i++){
        threads[i] = std::thread(
            [&data, &src_data, rows, cols](std::size_t start_idx, std::size_t end_idx){
                std::size_t idx = 0;
				for(std::size_t ti = start_idx; ti < end_idx; ti++){
		    		for(std::size_t tj = start_idx; tj != rows; tj++){
                        data[idx++] = src_data[tj * cols + ti]; 
	    			}
		    	}
            }, i * block_size , std::min(cols, (i + 1) * block_size)
        );
    }

    for(auto& thread : threads){
        thread.join();
    }

    mat.set_shallow_data(data, cols, rows);
}


template class MatrixHelper<int>;
template class MatrixHelper<float>;
template class MatrixHelper<double>;

}//namespace algebra
}//namespace abcdl
