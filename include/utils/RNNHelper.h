
/***********************************************
 * Author: Jun Jiang - jiangjun4@sina.com
 * Create: 2017-06-23 15:40
 * Last modified : 2017-06-23 15:40
 * Filename      : RNNHelper.h
 * Description   : Read & generate one-hot 
                   word vector 
 **********************************************/
#pragma once

#include <map>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <stdlib.h>
#include "algebra/Matrix.h"
#include "utils/Log.h"
#include "utils/StringHelper.h"


namespace abcdl{
namespace utils{

class RNNHelper{
public:
	RNNHelper(size_t feature_dim){
		_feature_dim = feature_dim;
	}

	bool read_word2index(const std::string& data_file, std::map<std::string, size_t>& dict);

	bool read_seq_data(const std::string& data_file,
					  std::vector<abcdl::algebra::Mat*>& train_seq_data,
					  const std::string& label_file,
					  std::vector<abcdl::algebra::Mat*>& train_seq_label,
                      int limit = -1);
private:
	bool read_data(const std::string& data_file,
                   std::vector<abcdl::algebra::Mat*>& mat,
                   int limit = -1);

private:
	size_t _feature_dim;

};//class RNNHelper

	
bool RNNHelper::read_word2index(const std::string& data_file, std::map<std::string, size_t>& dict){
	std::ifstream in_file(data_file.data());;
	if(!in_file.is_open()){
		LOG(FATAL) << "Open file  failed:" << data_file.c_str();
		return false;
	}

    abcdl::utils::StringHelper helper;
	std::string data;
	while(getline(in_file, data)){
        auto line_data = helper.split(data, "\t");
        if(line_data.size() == 2){
			size_t value = helper.str2int(line_data[1]);
			if(value < _feature_dim){
				dict.insert(std::map<std::string, size_t>::value_type(line_data[0], value));
			}
        }
	}
	in_file.close();
	return true;
}

bool RNNHelper::read_seq_data(const std::string& data_file,
				   		      std::vector<abcdl::algebra::Mat*>& train_seq_data,
					  	 	  const std::string& label_file,
					  		  std::vector<abcdl::algebra::Mat*>& train_seq_label,
                             int limit){
    return read_data(data_file, train_seq_data, limit) &&read_data(label_file, train_seq_label, limit);
}

bool RNNHelper::read_data(const std::string& data_file,
                          std::vector<abcdl::algebra::Mat*>& mat,
                          int limit){
	std::ifstream in_file(data_file.c_str());
	if(!in_file.is_open()){
		LOG(FATAL) << "Open file failed:" << data_file.c_str();
		return false;
	}
		
	std::string data;
    abcdl::utils::StringHelper helper;
	while(getline(in_file, data)){
		auto line_data = helper.split(data, "\t");
        size_t rows = line_data.size();
        auto mat_data = new abcdl::algebra::Mat(rows, _feature_dim);
        for(size_t i = 0; i != rows ; i++){
            mat_data->set_data(1, i, helper.str2int(line_data[i]));
        }
        mat.push_back(mat_data);
        if(limit > 0 && (int)mat.size() >= limit){
            break;
        }
		in_file.close();
	}
	return true;
}


}//namespace
}//namespace abcdl
