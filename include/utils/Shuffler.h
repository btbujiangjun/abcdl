/*********************************************
* Author: Jun Jiang - jiangjun4@sina.com
* Created: 2017-08-19 16:33
* Last modified: 2017-08-19 17:04
* Filename: Shuffler.h
* Description: matrix shuffle
**********************************************/
#pragma once

#include <random>
#include <vector>
#include <ctime>

namespace abcdl{
namespace utils{

class Shuffler{
public:
    Shuffler(const size_t size);

    ~Shuffler(){ _shuffler_idx.clear();}

    void shuffle();

    size_t get(const size_t row_id) const {return _shuffler_idx[row_id];}

private:
    size_t _size;
    std::vector<size_t> _shuffler_idx;
};//class Shuffler


Shuffler::Shuffler(const size_t size){
    _size = size;
    for(size_t i = 0; i != _size; i++){
        _shuffler_idx.push_back(i);
    }
}

void Shuffler::shuffle(){

    std::random_device rd;
    size_t random_idx, value;

    for(size_t i = _size - 1; i != 0 ; i--){
        random_idx = rd() % i;
        value =  _shuffler_idx[random_idx];

        _shuffler_idx[random_idx] = _shuffler_idx[i];
        _shuffler_idx[i] = value;
    }

}

}//namespace utils
}//namespace abcdl
