/*********************************************
* Author: Jun Jiang - jiangjun4@sina.com
* Created: 2017-06-01 11:22
* Last modified: 2017-06-01 11:22
* Filename: TestRNN.cpp
* Description: 
**********************************************/

#include "rnn/RNN.h"
#include "utils/Log.h"
#include "utils/RNNHelper.h"
#include "utils/MnistHelper.h"

int main(int argc, char** argv){
    abcdl::utils::log::set_min_log_level(abcdl::utils::log::INFO);
    abcdl::utils::log::initialize_log(argc, argv);

    abcdl::rnn::RNN rnn(28, 100);
    rnn.set_epoch(50);
   
    const int train_size = 100;
    const int test_size = 10;
    abcdl::utils::FashionMnistReader<real> helper("data");
    std::vector<abcdl::algebra::Mat*> train_data;
    std::vector<abcdl::algebra::Mat*> train_label;
    helper.read_train_images(&train_data, train_size);
    helper.read_train_vec_labels(&train_label, train_size);

    LOG(INFO) << "RNN Start training....";
    LOG(INFO) << "training size:" << train_data.size();
    rnn.train(train_data, train_label);
	
    for(auto&& d : train_data){
		delete d;
	}
	train_data.clear();
	
	for(auto&& d : train_label){
		delete d;
	}
	train_label.clear();
/*
	abcdl::utils::RNNHelper helper(8000);
	
	//std::map<std::string, size_t> map;
	//helper.read_word2index("data/rnn/word_to_index", map);

    const size_t sample_size = 1000;
	std::vector<abcdl::algebra::Mat*> data_seq_data;
	std::vector<abcdl::algebra::Mat*> data_seq_label;
	if(!helper.read_seq_data("data/rnn/train_seq_data", data_seq_data, "data/rnn/train_seq_label", data_seq_label, sample_size)){
		return -1;
	}

    LOG(INFO) << "RNN Start training....";

    abcdl::rnn::RNN rnn(8000, 100);
    rnn.set_epoch(50);
    rnn.load_model("./data/rnn_seq.model");

    LOG(INFO) << "training size:" << data_seq_data.size();
    rnn.train(data_seq_data, data_seq_label);

	for(auto&& d : data_seq_data){
		delete d;
	}
	data_seq_data.clear();
	
	for(auto&& d : data_seq_label){
		delete d;
	}
	data_seq_label.clear();

*/

}
