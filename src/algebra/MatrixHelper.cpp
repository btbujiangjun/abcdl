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
Matrix<T> MatrixHelper<T>::dot(const Matrix<T>& mat_a, const Matrix<T>& mat_b){
    Matrix<T> mat;
    dot(mat, mat_a, mat_b);
    return mat;
}

template<class T>
void MatrixHelper<T>::dot(Matrix<T>& mat,
						  const Matrix<T>& mat_a,
						  const Matrix<T>& mat_b){
    size_t row_a = mat_a.rows();
    size_t col_a = mat_a.cols();
    size_t row_b = mat_b.rows();
    size_t col_b = mat_b.cols();

    CHECK(col_a == row_b);

    T* data   = new T[row_a * col_b];
    T* data_a = mat_a.data();
    T* data_b = mat_b.data();

    size_t size = row_a * col_b * col_a;
    size_t num_thread = po.get_num_thread(size, po.get_block_size(size));
    size_t block_size = row_a / num_thread;
    if(row_a % num_thread != 0){
        block_size += 1;
    }

    std::vector<std::thread> threads(num_thread);
    for(size_t i = 0; i != num_thread; i++){
        threads[i] = std::thread(
            [&data, &data_a, &data_b, &col_a, &col_b](size_t start_idx, size_t end_idx){
				for(size_t ti = start_idx; ti < end_idx; ti++){
					size_t a_init_idx = ti * col_a;
					for(size_t tj = 0; tj != col_b; tj++){
						T value = 0;
						size_t a_idx = a_init_idx;
						for(size_t tk = 0; tk != col_a; tk++){
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
Matrix<T> MatrixHelper<T>::outer(const Matrix<T>& mat_a, const Matrix<T>& mat_b){
	Matrix<T> mat;
	outer(mat, mat_a, mat_b);
	return mat;
}

template<class T>
void MatrixHelper<T>::outer(Matrix<T>& mat,
							const Matrix<T>& mat_a,
							const Matrix<T>& mat_b){
    size_t size_a = mat_a.get_size();
    size_t size_b = mat_b.get_size();
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
    if(mat.get_size() != mat_a.get_size()){
        mat.reset(0, mat_a.rows(), mat_a.cols());
    }
    po.parallel_mul2one_copy<T>(mat.data(), mat_a.data(), mat_a.get_size(), exponent, lamda);
}

template<class T>
void MatrixHelper<T>::log(Matrix<T>& mat, const Matrix<T>& mat_a){
    auto lamda = [](T* a, const T& b){ *a = std::log(b);};
    if(mat.get_size() == mat_a.get_size()){
        mat.reset(0, mat_a.rows(), mat_a.cols());
    }
    po.parallel_mul2one_copy<T>(mat.data(), mat_a.data(), mat_a.get_size(), lamda);
}

template<class T>
void MatrixHelper<T>::exp(Matrix<T>& mat, const Matrix<T>& mat_a){
    auto lamda = [](T* a, const T& b){ *a = std::exp(std::min(b, (T)EXP_MAX));};
    if(mat.get_size() != mat_a.get_size()){
        mat.reset(0, mat_a.rows(), mat_a.cols());
    }
    po.parallel_mul2one_copy<T>(mat.data(), mat_a.data(), mat_a.get_size(), lamda);
}

template<class T>
void MatrixHelper<T>::sqrt(Matrix<T>& mat, const Matrix<T>& mat_a){
    auto lamda = [](T* a, const T& b){ *a = std::sqrt(b);};
    if(mat.get_size() != mat_a.get_size()){
        mat.reset(0, mat_a.rows(), mat_a.cols());
    }
    po.parallel_mul2one_copy<T>(mat.data(), mat_a.data(), mat_a.get_size(), lamda);
}

template<class T>
void MatrixHelper<T>::sigmoid(Matrix<T>& mat, const Matrix<T>& mat_a){
    auto lamda = [](T* a, const T& b){ *a = 1 / (1 + std::exp(-(std::min((T)SIGMOID_MAX, std::max(b, (T)SIGMOID_MIN)))));};
    if(mat.get_size() != mat_a.get_size()){
        mat.reset(0, mat_a.rows(), mat_a.cols());
    }
    po.parallel_mul2one_copy<T>(mat.data(), mat_a.data(), mat_a.get_size(), lamda);
}

template<class T>
void MatrixHelper<T>::sigmoid_derivative(Matrix<T>& mat, const Matrix<T>& mat_a){
    auto lamda = [](T* a, const T& b){ *a = b * (1 - b);};
    if(mat.get_size() != mat_a.get_size()){
        mat.reset(0, mat_a.rows(), mat_a.cols());
    }
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
    if(mat.get_size() != mat_a.get_size()){
        mat.reset(0, mat_a.rows(), mat_a.cols());
    }
    po.parallel_mul2one_copy<T>(mat.data(), mat_a.data(), mat_a.get_size(), lamda);
}

template<class T>
void MatrixHelper<T>::tanh_derivative(Matrix<T>& mat, const Matrix<T>& mat_a){
    auto lamda = [](T* a, const T& b){ T tanh = std::exp(std::min((T)EXP_MAX, -2 * b)); *a = 1 - tanh * tanh;};
    if(mat.get_size() != mat_a.get_size()){
        mat.reset(0, mat_a.rows(), mat_a.cols());
    }
    po.parallel_mul2one_copy<T>(mat.data(), mat_a.data(), mat_a.get_size(), lamda);
}

template<class T>
void MatrixHelper<T>::relu(Matrix<T>& mat, const Matrix<T>& mat_a){
    auto lamda = [](T* a, const T& b){ if(b < 0) {*a = 0;} };
    if(&mat != &mat_a){
        mat.set_data(mat_a.data(), mat_a.rows(), mat_a.cols());
    }
    po.parallel_mul2one_copy<T>(mat.data(), mat_a.data(), mat_a.get_size(), lamda);
}

template<class T>
void MatrixHelper<T>::relu_derivative(Matrix<T>& mat, const Matrix<T>& mat_a){
    auto lamda = [](T* a, const T& b){ if(b > 0) {*a = 1;} };
    if(&mat != &mat_a){
        mat.reset(0, mat_a.rows(), mat_a.cols());
    }
    po.parallel_mul2one_copy<T>(mat.data(), mat_a.data(), mat_a.get_size(), lamda);
}

template<class T>
void MatrixHelper<T>::leaky_relu(Matrix<T>& mat, const Matrix<T>& mat_a){
    auto lamda = [](T* a, const T& b){ if(b < 0) {*a = (T)(0.01 * b);} };
    if(&mat != &mat_a){
        mat.set_data(mat_a.data(), mat_a.rows(), mat_a.cols());
    }
    po.parallel_mul2one_copy<T>(mat.data(), mat_a.data(), mat_a.get_size(), lamda);
}

template<class T>
void MatrixHelper<T>::leaky_relu_derivative(Matrix<T>& mat, const Matrix<T>& mat_a){
    auto lamda = [](T* a, const T& b){ if(b >= 0){*a = (T)1;} else{*a = (T)0.01;} };
    if(&mat != &mat_a){
        mat.reset(0, mat_a.rows(), mat_a.cols());
    }
    po.parallel_mul2one_copy<T>(mat.data(), mat_a.data(), mat_a.get_size(), lamda);
}

template<class T>
void MatrixHelper<T>::elu(Matrix<T>& mat, const Matrix<T>& mat_a){
    auto lamda = [](T* a, const T& b){ if(b >= 0) {*a = b;} else{*a = std::exp(std::min((T)EXP_MAX, b)) - 1;} };
    if(&mat != &mat_a){
        mat.reset(0, mat_a.rows(), mat_a.cols());
    }
    po.parallel_mul2one_copy<T>(mat.data(), mat_a.data(), mat_a.get_size(), lamda);
}

template<class T>
void MatrixHelper<T>::elu_derivative(Matrix<T>& mat, const Matrix<T>& mat_a){
    auto lamda = [](T* a, const T& b){ if(b >= 0) {*a = 1;} else{*a = std::exp(std::min((T)EXP_MAX, b));} };
    if(&mat != &mat_a){
        mat.reset(0, mat_a.rows(), mat_a.cols());
    }
    po.parallel_mul2one_copy<T>(mat.data(), mat_a.data(), mat_a.get_size(), lamda);
}

template<class T>
void MatrixHelper<T>::expand(Matrix<T>& result,
                             const Matrix<T>& mat,
                             const size_t row_dim,
                             const size_t col_dim){

    CHECK(row_dim * col_dim > 1);
   
    size_t col_a = mat.cols();
    size_t row   = mat.rows() * row_dim;
    size_t col   = col_a * col_dim;
    size_t size  = row * col;

    size_t num_thread = po.get_num_thread(size, po.get_block_size(size));
    size_t block_size = row / num_thread;
    if(row % num_thread != 0){
        block_size += 1;
    }

    std::vector<std::thread> threads(num_thread);
    T* data = mat.data();
    T* new_data = new T[size];
    memset(new_data, 0, sizeof(T) * size);

    for(size_t i = 0; i != num_thread; i++){
        threads[i] = std::thread(
            [&data, &new_data,&col, &col_a, &row_dim, &col_dim](size_t start_idx, size_t end_idx){
				for(size_t ti = start_idx; ti < end_idx; ti++){
					for(size_t tj = 0; tj != col; tj++){
                        new_data[ti * col + tj] = data[ti / row_dim * col_a + tj / col_dim];
					}
				}
            }, i * block_size , std::min(row, (i + 1) * block_size)
        );
    }

    for(auto& thread : threads){
        thread.join();
    }

    result.set_shallow_data(new_data, row, col);
}

template<class T>
bool MatrixHelper<T>::convn(Matrix<T>& result,
                            const Matrix<T>& mat,
                            const Matrix<T>& kernal,
                            const size_t stride,
                            const Convn_type type){
    size_t rows       = mat.rows();
    size_t cols       = mat.cols();
    size_t kernal_row = kernal.rows();
    size_t kernal_col = kernal.cols();

    size_t data_row;
    size_t data_col;

    T* data;
    T* new_data;
    T* src_data = mat.data();

    if(type == FULL && kernal_row * kernal_col != 1){
        data_row = mat.rows() + 2 * (kernal_row - 1);
        data_col = mat.cols() + 2 * (kernal_col - 1);

        data = new T[data_row * data_col];
        memset(data, 0, sizeof(T) * data_row * data_col);

        //padding 0, (kernal_row - 1)*(kernal_col - 1)
        for(size_t i = 0; i != rows; i++){
            memcpy(&data[(i + kernal_row - 1) * data_col + kernal_col - 1], &src_data[i * cols], sizeof(T)* cols);
        }
    }else if(type == VALID){
        data_row = rows;
        data_col = cols;
        data     = src_data;
        
        if(data_row < kernal_row || data_col < kernal_col){
            LOG(FATAL) << "Convn error: kernal size large than mat.";
            return false;
        }
    }else{
        //todo
        LOG(FATAL) << "todo:not support SAME type";
        return false;
    }


    size_t conv_row = (data_row - kernal_row) % stride == 0 ? (data_row - kernal_row) / stride + 1 : (data_row - kernal_row) / stride + 2;
    size_t conv_col = (data_col - kernal_col) % stride == 0 ? (data_col - kernal_col) / stride + 1 : (data_col - kernal_col) / stride + 2;

    new_data = new T[conv_row * conv_col];
    T* kernal_data = kernal.data();
    
    size_t size = conv_row * conv_col * kernal_row * kernal_col;
    size_t num_thread = po.get_num_thread(size, po.get_block_size(size));
    size_t block_size = conv_row / num_thread;
    if(conv_row % num_thread != 0){
        block_size += 1;
    }

    std::vector<std::thread> threads(num_thread);
    for(size_t i = 0; i != num_thread; i++){
        threads[i] = std::thread(
            [&data, &new_data, &kernal_data, &data_row, &data_col, &kernal_row, &kernal_col, &conv_col, &stride](size_t start_idx, size_t end_idx){
				for(size_t ti = start_idx; ti < end_idx; ti++){
                    for(size_t tj = 0; tj != conv_col; tj++){
                        T sum = 0;
                        for(size_t k_i = 0; k_i != kernal_row; k_i++){
                           size_t row = ti * stride + k_i;
                            for(size_t k_j = 0; k_j != kernal_col; k_j++){
                                size_t col = tj * stride + k_j;
                                //skip out of range, in other word, fill 0
                                if(row < data_row && col < data_col){
                                    T a = data[row * data_col + col];
                                    T b = kernal_data[k_i * kernal_col + k_j];
                                    if(a != 0 && b != 0){
                                        sum += a * b;
                                    }
                                }
                            }
                        }
                        new_data[ti * conv_col + tj] = sum;
                    }
				}
            }, i * block_size , std::min(conv_row, (i + 1) * block_size)
        );
    }

    for(auto& thread : threads){
        thread.join();
    }

    result.set_shallow_data(new_data, conv_row, conv_col);

    if(type == abcdl::algebra::FULL){
    	delete[] data;
    }

    return true;
}

template<class T>
void MatrixHelper<T>::zero_like(Matrix<T>& mat, const Matrix<T>& mat_a){
	mat.reset(0, mat_a.rows(), mat_a.cols()); 
}


template<class T>
void MatrixHelper<T>::transpose(Matrix<T>& mat, const Matrix<T>& mat_a){
	size_t rows = mat_a.rows();
	size_t cols = mat_a.cols();
	if(rows == 1 || cols == 1){
        if(&mat != &mat_a){
    		mat.set_data(mat_a.data(), rows, cols);
        }
		mat.reshape(cols, rows);
        return ;
	}

    T* src_data       = mat_a.data();
    T* data           = new T[mat_a.get_size()];
    size_t num_thread = po.get_num_thread(mat_a.get_size(), po.get_block_size(mat_a.get_size()));
    size_t block_size = cols / num_thread;
    if(cols % num_thread != 0){
        block_size += 1;
    }
        
    std::vector<std::thread> threads(num_thread);
    for(size_t i = 0; i != num_thread; i++){
        threads[i] = std::thread(
            [&data, &src_data, rows, cols](size_t start_idx, size_t end_idx){
				for(size_t ti = start_idx; ti < end_idx; ti++){
		    		for(size_t tj = 0; tj != rows; tj++){
                        data[ti * rows + tj] = src_data[tj * cols + ti];
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
template class MatrixHelper<size_t>;

}//namespace algebra
}//namespace abcdl
