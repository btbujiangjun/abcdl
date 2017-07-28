/***********************************************
 * Author: Jun Jiang - jiangjun4@sina.com
 * Create: 2017-07-26 13:56
 * Last modified : 2017-07-26 13:56
 * Filename      : ParallelOperator.h
 * Description   : Parallel Operate by multithread 
 **********************************************/

#ifndef _ABCDL_UTILS_PARALLEL_OPERATOR_H_
#define _ABCDL_UTILS_PARALLEL_OPERATOR_H_

#include <vector>
#include <thread>
#include <functional>

namespace abcdl{
namespace utils{

class ParallelOperator{
public:
    ParallelOperator(){
        _num_thread = std::thread::hardware_concurrency();
    }

    ParallelOperator(const uint num_thread){
        _num_thread = num_thread;
    }

    template<class T>
    void parallel_mul2one(T* op1,
                          const uint num_op1,
                          const std::function<void(T*)> &f){
        uint block_size = get_block_size(num_op1);
        uint num_thread = get_num_thread(num_op1, block_size);
        std::vector<std::thread> threads(num_thread);

        for(uint i = 0; i != num_thread; i++){
            threads[i] = std::thread(
                [&op1, &f](uint start_idx, uint end_idx){
                    for(uint ti = start_idx; ti != end_idx; ti++){
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
                          const uint num_op1,
                          const T& op2,
                          const std::function<void(T*, const T&)> &f){
        uint block_size = get_block_size(num_op1);
        uint num_thread = get_num_thread(num_op1, block_size);
        std::vector<std::thread> threads(num_thread);

        for(uint i = 0; i != num_thread; i++){
            threads[i] = std::thread(
                [&op1, &op2, &f](uint start_idx, uint end_idx){
                    for(uint ti = start_idx; ti != end_idx; ti++){
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
    void parallel_mul2one_copy(const T* op1,
                               const uint num_op1,
                               T* result_data,
                               const std::function<void(T*, const T&)> &f){
        uint block_size = get_block_size(num_op1);
        uint num_thread = get_num_thread(num_op1, block_size);
        std::vector<std::thread> threads(num_thread);

        for(uint i = 0; i != num_thread; i++){
            threads[i] = std::thread(
                [&result_data, &op1, &f](uint start_idx, uint end_idx){
                    for(uint ti = start_idx; ti != end_idx; ti++){
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
    void parallel_mul2one_copy(const T* op1,
                               const uint num_op1,
                               const T& op2,
                               T* result_data,
                               const std::function<void(T*, const T&, const T&)> &f){
        uint block_size = get_block_size(num_op1);
        uint num_thread = get_num_thread(num_op1, block_size);
        std::vector<std::thread> threads(num_thread);

        for(uint i = 0; i != num_thread; i++){
            threads[i] = std::thread(
                [&result_data, &op1, &f, &op2](uint start_idx, uint end_idx){
                    for(uint ti = start_idx; ti != end_idx; ti++){
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
                          const uint num_op1,
                          const T* op2,
                          const std::function<void(T*, const T&)> &f){

        uint block_size = get_block_size(num_op1);
        uint num_thread = get_num_thread(num_op1, block_size);
        std::vector<std::thread> threads(num_thread);

        for(uint i = 0; i != num_thread; i++){
            threads[i] = std::thread(
                [&op1, &op2, &f](uint start_idx, uint end_idx){
                    for(uint ti = start_idx; ti != end_idx; ti++){
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
    void parallel_reduce(const T* op1,
                         const uint num_op1,
                         T* op2,
                         const std::function<void(const T&, T*)> &f){
        uint block_size = get_block_size(num_op1);
        uint num_thread = get_num_thread(num_op1, block_size);
        std::vector<std::thread> threads(num_thread);

        for(uint i = 0; i != num_thread; i++){
            threads[i] = std::thread(
                [&op1, &op2, &f](uint start_idx, uint end_idx){
                    for(uint ti = start_idx; ti != end_idx; ti++){
                        f(op1[ti], &op2);
                    }
                }, i * block_size, std::min(num_op1, (i + 1) * block_size)
            );
        }

        for(auto& thread : threads){
            thread.join();
        }
    }


private:
    uint _num_thread;
    uint _min_block_size = 1000;

    uint get_block_size(const uint size) const{
        uint block_size = size / _num_thread;
        if( size % _num_thread != 0){
            block_size += 1;
        }
        return block_size < _min_block_size ? std::min(size, _min_block_size) : block_size;
    }

    uint get_num_thread(const uint size, const uint block_size) const{
        if(block_size >= _min_block_size){
            return _num_thread;
        }
        uint num_thread = size / block_size;
        if(size % block_size != 0){
            num_thread += 1;
        }
        return num_thread;
    }
};//class ParallelOperator

}//namespace utils
}//namespace abcdl

#endif //_ABCDL_UTILS_PARALLEL_OPERATOR_H_
