/***********************************************
 * Author: Jun Jiang - jiangjun4@sina.com
 * Create: 2017-09-05 16:04
 * Last modified : 2017-09-06 14:42
 * Filename	  : Layer.cpp
 * Description   : RNN network Layer 
 **********************************************/
#include "rnn/Layer.h"

namespace abcdl{
namespace rnn{

void Layer::farward(const abcdl::algebra::Mat& train_seq_data,
					const abcdl::algebra::Mat& weight,
					const abcdl::algebra::Mat& pre_weight,
					const abcdl::algebra::Mat& act_weight,
					abcdl::algebra::Mat& state,
					abcdl::algebra::Mat& activation){
	
	size_t seq_rows = train_seq_data.rows();
	size_t seq_cols = train_seq_data.cols();
	state.reset(0, seq_rows, _hidden_dim);
	activation.reset(0, seq_rows, seq_cols);

	abcdl::algebra::Mat s_t;
	for(size_t t = 0; t != seq_rows; t++){
		//s[t] = tanh(U*x[t] + W*s[t-1])
		//o[t] = softmax(V* s[t])
		if(t > 0){
			s_t = (helper.dot(weight, train_seq_data.get_row(t).Ts()) + helper.dot(pre_weight, state.get_row(t - 1).Ts())).tanh();
		}else{
			s_t = helper.dot(weight, train_seq_data.get_row(t).Ts()).tanh();
		}

		state.set_row(t, s_t.Ts());
		activation.set_row(t, helper.dot(act_weight, s_t).softmax().Ts());	
	}
}

void Layer::backward(const abcdl::algebra::Mat& train_seq_data,
					 const abcdl::algebra::Mat& train_seq_label,
					 abcdl::algebra::Mat& weight,
					 abcdl::algebra::Mat& pre_weight,
					 abcdl::algebra::Mat& act_weight,
					 const abcdl::algebra::Mat& state,
					 const abcdl::algebra::Mat& activation,
					 abcdl::algebra::Mat& derivate_weight,
					 abcdl::algebra::Mat& derivate_pre_weight,
					 abcdl::algebra::Mat& derivate_act_weight){

	//todo cost function
	abcdl::algebra::Mat derivate_output;
	_cost->delta(derivate_output, activation,train_seq_label);
	
	size_t seq_size = train_seq_data.rows();
    auto now = []{return std::chrono::system_clock::now();};
    auto start_time = now();

	for(long int t = seq_size - 1; t >= 0 ; t--){
		auto derivate_output_t = derivate_output.get_row(t).Ts();
		auto state_t = state.get_row(t).Ts();

        //update derivate_act_weight
		derivate_output_t.outer(state_t);
		derivate_act_weight += derivate_output_t;

        //calc derivate_t
		abcdl::algebra::Mat state_derivate;
		helper.sigmoid_derivative(state_derivate, state_t);
		auto derivate_t = helper.dot(act_weight.Ts(), derivate_output_t.Ts()) * state_derivate;

		//back_propagation steps
		for(size_t step = 0; step < _bptt_truncate && (int)step <= t; step++){
			size_t bptt_step = t - step;

    		printf("Backpropagation step t=%ld bptt step=%ld\n", t, bptt_step);

			abcdl::algebra::Mat derivate_state_t;
			if(bptt_step > 0){
				state.get_row(&derivate_state_t, bptt_step -1);
			}else{
				derivate_state_t.reset(0, state.cols(), 1);
			}

            //update derivate_pre_weight
			derivate_pre_weight += helper.outer(derivate_t, derivate_state_t.Ts());
			
            //update derivate_weight
			auto train_data_t = train_seq_data.get_row(bptt_step).Ts();
            auto derivate_weight_t = helper.dot(derivate_weight, train_data_t) + derivate_t;

            size_t idx = train_data_t.argmax(0, abcdl::algebra::Axis_type::ROW);
            derivate_weight.set_col(idx, derivate_weight.get_col(idx) + derivate_weight_t);

			//update delta
			if(bptt_step > 0){
				helper.sigmoid_derivative(derivate_state_t, derivate_state_t);
				derivate_t = pre_weight.Ts().dot(derivate_t) * derivate_state_t;
			}
		}
	}
    printf("thread[%lu][%ld-%ld][%ld].\n", std::this_thread::get_id(), start_time, now(), std::chrono::duration_cast<std::chrono::milliseconds>(now() - start_time).count());
}


}//namespace rnn
}//namespace abcdl
