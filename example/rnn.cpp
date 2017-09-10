/*********************************************
* Author: Jun Jiang - jiangjun4@sina.com
* Created: 2017-06-01 11:22
* Last modified: 2017-06-01 11:22
* Filename: TestRNN.cpp
* Description: 
**********************************************/

#include "utils/RNNHelper.h"
#include "rnn/RNN.h"

int main(int argc, char** argv){
	std::map<std::string, size_t> map;
	abcdl::utils::RNNHelper helper(8000);
	helper.read_word2index("data/rnn/word_to_index", map);

	std::vector<abcdl::algebra::Mat*> data_seq_data;
	std::vector<abcdl::algebra::Mat*> data_seq_label;
	helper.read_seqdata("data/rnn/train_seq_data", &data_seq_data, "data/rnn/train_seq_label", &data_seq_label, 1000);

    printf("Start training....\n");

    abcdl::rnn::RNN rnn(8000, 100);
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
