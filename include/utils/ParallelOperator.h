/***********************************************
 * Author: Jun Jiang - jiangjun4@sina.com
 * Create: 2017-07-26 13:56
 * Last modified : 2017-07-26 13:56
 * Filename      : ParallelOperator.h
 * Description   : Parallel Operate by multithread 
 **********************************************/

#ifndef _ABCDL_UTILS_PARALLEL_OPERATOR_H_
#define _ABCDL_UTILS_PARALLEL_OPERATOR_H_

#include <cmath>
#include <vector>
#include <thread>
#include <functional>
#include "utils/TypeDef.h"

namespace abcdl{
namespace utils{

class ParallelOperator{
public:
    ParallelOperator(){
        _num_thread = std::thread::hardware_concurrency();
    }

    ParallelOperator(const std::size_t num_thread){
        _num_thread = num_thread;
    }

    template<class T>
    void parallel_mul2one(T* op1,
                          const std::size_t num_op1,
                          const std::function<void(T*)> &f){
        std::size_t block_size = get_block_size(num_op1);
        std::size_t num_thread = get_num_thread(num_op1, block_size);
        std::vector<std::thread> threads(num_thread);

        for(std::size_t i = 0; i != num_thread; i++){
            threads[i] = std::thread(
                [&op1, &f](std::size_t start_idx, std::size_t end_idx){
                    for(std::size_t ti = start_idx; ti != end_idx; ti++){
                        f(&op1[ti]);
                    }
                }, i * block_size, std::min(num_op1, (i + 1) * block_size)
            );
        }

        for(auto& thread : threads){
            thread.join();
        }
    }

    template<class T>
    void parallel_mul2one(T* op1,
                          const std::size_t num_op1,
                          const T& op2,
                          const std::function<void(T*, const T&)> &f){
        std::size_t block_size = get_block_size(num_op1);
        std::size_t num_thread = get_num_thread(num_op1, block_size);
        std::vector<std::thread> threads(num_thread);

        for(std::size_t i = 0; i != num_thread; i++){
            threads[i] = std::thread(
                [&op1, &op2, &f](std::size_t start_idx, std::size_t end_idx){
                    for(std::size_t ti = start_idx; ti != end_idx; ti++){
                        f(&op1[ti], op2);
                    }
                }, i * block_size, std::min(num_op1, (i + 1) * block_size)
            );
        }

        for(auto& thread : threads){
            thread.join();
        }
    }

    template<class T>
    void parallel_mul2one_copy(T* result_data,
                               const T* op1,
                               const std::size_t num_op1,
                               const std::function<void(T*, const T&)> &f){
        std::size_t block_size = get_block_size(num_op1);
        std::size_t num_thread = get_num_thread(num_op1, block_size);
        std::vector<std::thread> threads(num_thread);

        for(std::size_t i = 0; i != num_thread; i++){
            threads[i] = std::thread(
                [&result_data, &op1, &f](std::size_t start_idx, std::size_t end_idx){
                    for(std::size_t ti = start_idx; ti != end_idx; ti++){
                        f(&result_data[ti], op1[ti]);
                    }
                }, i * block_size, std::min(num_op1, (i + 1) * block_size)
            );
        }

        for(auto& thread : threads){
            thread.join();
        }
    }

    template<class T>
    void parallel_mul2one_copy(T* result_data,
                               const T* op1,
                               const std::size_t num_op1,
                               const T& op2,
                               const std::function<void(T*, const T&, const T&)> &f){
        std::size_t block_size = get_block_size(num_op1);
        std::size_t num_thread = get_num_thread(num_op1, block_size);
        std::vector<std::thread> threads(num_thread);

        for(std::size_t i = 0; i != num_thread; i++){
            threads[i] = std::thread(
                [&result_data, &op1, &f, &op2](std::size_t start_idx, std::size_t end_idx){
                    for(std::size_t ti = start_idx; ti != end_idx; ti++){
                        f(&result_data[ti], op1[ti], op2);
                    }
                }, i * block_size, std::min(num_op1, (i + 1) * block_size)
            );
        }

        for(auto& thread : threads){
            thread.join();
        }
    }

    template<class T>
    void parallel_mul2mul(T* op1,
                          const std::size_t num_op1,
                          const T* op2,
                          const std::size_t num_op2,
                          const std::function<void(T*, const T&)> &f){

        if(num_op1 != num_op2){
            //todo 
        }

        std::size_t block_size = get_block_size(num_op1);
        std::size_t num_thread = get_num_thread(num_op1, block_size);
        std::vector<std::thread> threads(num_thread);

        for(std::size_t i = 0; i != num_thread; i++){
            threads[i] = std::thread(
                [&op1, &op2, &f](std::size_t start_idx, std::size_t end_idx){
                    for(std::size_t ti = start_idx; ti != end_idx; ti++){
                        f(&op1[ti], op2[ti]);
                    }
                }, i * block_size, std::min(num_op1, (i + 1) * block_size)
            );
        }

        for(auto& thread : threads){
            thread.join();
        }
    }

    template<class T>
    void parallel_mul2mul_cross(T* result_data,
                                const T* op1,
                                const std::size_t num_op1,
                                const T* op2,
                                const std::size_t num_op2,
                                const std::function<void(T*, const T&, const T&)> &f){

        std::size_t block_size = get_block_size(num_op1 * num_op2);
        std::size_t num_thread = get_num_thread(num_op1 * num_op2, block_size);
        std::vector<std::thread> threads(num_thread);
        block_size = num_op1 / num_thread;
        if(num_op1 % num_thread != 0){
            block_size += 1;
        }

        for(std::size_t i = 0; i != num_thread; i++){
            threads[i] = std::thread(
                [&result_data, &op1, &op2, num_op2, &f](std::size_t start_idx, std::size_t end_idx){
                    std::size_t idx = start_idx * num_op2;
                    for(std::size_t ti = start_idx; ti != end_idx; ti++){
                        for(std::size_t tj = 0; tj != num_op2; tj++){
                            f(&result_data[idx++], op1[ti], op2[tj]);
                        }
                    }
                }, i * block_size, std::min(num_op1, (i + 1) * block_size)
            );
        }

        for(auto& thread : threads){
            thread.join();
        }
    }

    template<class T>
    void parallel_reduce_mul2one(T* result_value,
                                 const T* op1,
                         		 const std::size_t num_op1,
                         		 const std::function<void(T*, const T&)> &f){
        std::size_t block_size = get_block_size(num_op1);
        std::size_t num_thread = get_num_thread(num_op1, block_size);
        std::vector<std::thread> threads(num_thread);
		T* data = new T[num_thread];

        for(std::size_t i = 0; i != num_thread; i++){
            threads[i] = std::thread(
                [&op1, &result_value, &data, i, &f](std::size_t start_idx, std::size_t end_idx){
					for(std::size_t ti = start_idx; ti != end_idx; ti++){
            			if(ti == start_idx){
							data[i] = op1[ti];
						}else{
    						f(&data[i], op1[ti]);
                        }
                    }
					f(result_value, data[i]);
                }, i * block_size, std::min(num_op1, (i + 1) * block_size)
            );
        }

        for(auto& thread : threads){
            thread.join();
        }
		delete[] data;
    }

    template<class T>
    void parallel_reduce_boolean(bool* result_value,
                                  const T* op1,
                             	  const std::size_t num_op1,
                         		  const std::function<void(bool*, const T&)> &f){
        std::size_t block_size = get_block_size(num_op1);
        std::size_t num_thread = get_num_thread(num_op1, block_size);
        std::vector<std::thread> threads(num_thread);

        for(std::size_t i = 0; i != num_thread; i++){
            threads[i] = std::thread(
                [&result_value, &op1, &f](std::size_t start_idx, std::size_t end_idx){
					for(std::size_t ti = start_idx; ti != end_idx && *result_value; ti++){
    					f(result_value, op1[ti]);
                    }
                }, i * block_size, std::min(num_op1, (i + 1) * block_size)
            );
        }

        for(auto& thread : threads){
            thread.join();
        }
    }

    template<class T>
    void parallel_reduce_boolean(bool* result_value,
                                 const T* op1,
                            	 const std::size_t num_op1,
                                 const T* op2,
                            	 const std::size_t num_op2,
                         		 const std::function<void(bool*, const T&, const T&)> &f){
        if(num_op1 != num_op2){
            //todo diff size
        }
        std::size_t block_size = get_block_size(num_op1);
        std::size_t num_thread = get_num_thread(num_op1, block_size);
        std::vector<std::thread> threads(num_thread);

        for(std::size_t i = 0; i != num_thread; i++){
            threads[i] = std::thread(
                [&result_value, &op1, &op2, &f](std::size_t start_idx, std::size_t end_idx){
					for(std::size_t ti = start_idx; ti != end_idx && *result_value; ti++){
    					f(result_value, op1[ti], op2[ti]);
                    }
                }, i * block_size, std::min(num_op1, (i + 1) * block_size)
            );
        }

        for(auto& thread : threads){
            thread.join();
        }
    }

    inline std::size_t get_block_size(const std::size_t size) const{
        std::size_t block_size = size / _num_thread;
        if( size % _num_thread != 0){
            block_size += 1;
        }
        return block_size < _min_block_size ? std::min(size, _min_block_size) : block_size;
    }

    inline std::size_t get_num_thread(const std::size_t size, const std::size_t block_size) const{
        if(block_size >= _min_block_size){
            return _num_thread;
        }
        std::size_t num_thread = size / block_size;
        if(size % block_size != 0){
            num_thread += 1;
        }
        return num_thread;
    }

private:
    std::size_t _num_thread;
    std::size_t _min_block_size = 1000;
};//class ParallelOperator

}//namespace utils
}//namespace abcdl

#endif //_ABCDL_UTILS_PARALLEL_OPERATOR_H_
