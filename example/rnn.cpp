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

int main(int argc, char** argv){
    abcdl::utils::log::set_min_log_level(abcdl::utils::log::INFO);
    abcdl::utils::log::initialize_log(argc, argv);

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
}
