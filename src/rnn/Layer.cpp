/***********************************************
 * Author: Jun Jiang - jiangjun4@sina.com
 * Create: 2017-09-05 16:04
 * Last modified : 2017-09-05 16:05
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
			s_t = (helper.dot(weight, train_seq_data.get_row(t).Ts()) + helper.dot(pre_weight, state.get_row(t - 1).Ts())).tanh().Ts();
		}else{
			s_t = helper.dot(weight, train_seq_data.get_row(t).Ts()).tanh().Ts();
		}	
		state.set_row(t, s_t);

		activation.set_row(t, helper.dot(act_weight, s_t).softmax().Ts())	
	}
}

void Layer::backward(const abcdl::algebra::Mat& train_seq_data,
					 const abcdl::algebra::Mat& train_seq_label,
					 const abcdl::algebra::Mat& weight,
					 const abcdl::algebra::Mat& pre_weight,
					 const abcdl::algebra::Mat& act_weight,
					 abcdl::algebra::Mat& derivate_weight,
					 abcdl::algebra::Mat& derivate_pre_weight,
					 abcdl::algebra::Mat& derivate_act_weight){

    auto now = []{return std::chrono::system_clock::now();};
    auto start_time = now();

	farward(train_seq_data, weight, pre_weight, act_weight, state, derivate_output);
	derivate_output.subtract(train_seq_label);

	size_t seq_size = train_seq_data.rows();

	for(sizt_t t = seq_size - 1; t >= 0 ; t--){
		auto derivate_output_tt = derivate_output.get_row(t).Ts();
        //update derivate_act_weight
		dervate_act_weight += derivate_output_tt.outer(state.get_row(t).Ts());

        //calc derivate_t
		auto derivate_t = helper.dot(act_weight.Ts(), derivate_output_tt) * state.get_row(t).Ts().sigmoid_derivative();

		//back_propagation steps
		for(size_t step = 0; step < _bptt_truncate && (int)step <= t; step++){
			int bptt_step = t - step;

            if(debug){
    			printf("Backpropagation step t=%d bptt step=%d\n", t, bptt_step);
            }

			abcdl::algebra::Mat derivate_state_t;
			if(bptt_step > 0){
				state.get_row(&derivate_state_t, bptt_step -1);
			}else{
				derivate_state_t.reset(0, state.get_cols(), 1);
			}

            //update derivate_pre_weight
			derivate_pre_weight += helper.outer(derivate_t, derivate_state_t.Ts());
			
            //update derivate_weight
			train_seq_data.get_row_data(bptt_step, train_data_t);
            derivate_weight.clone(derivate_weight_t);

			derivate_weight_t.dot(train_data_t.transpose());
			derivate_weight_t.add(derivate_t);

            size_t idx = train_data_t.argmax(0, 1);
            derivate_weight.get_col_data(idx, derivate_weight_t_c);
            derivate_weight_t_c.add(derivate_weight_t);
            derivate_weight.set_col_data(idx, derivate_weight_t_c);

			//update delta
			if(bptt_step > 0){
				derivate_state_t.multiply(derivate_state_t);
				derivate_state_t.multiply(-1);
				derivate_state_t.add(1);

				pre_weight.clone(derivate_pre_weight_t);
				derivate_pre_weight_t.transpose().dot(derivate_t);
				derivate_pre_weight_t.multiply(derivate_state_t);
				derivate_pre_weight_t.clone(derivate_t);
			}
		}
	}

	delete derivate_output;
	delete derivate_output_t;
    delete derivate_weight_t;
    delete derivate_weight_t_c;
	delete derivate_t;
	delete derivate_pre_weight_t;
	delete derivate_state_t;
	delete train_data_t;

    auto end_time = now();
    printf("thread[%lu][%ld-%ld][%ld].\n", std::this_thread::get_id(), start_time, end_time, std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count());
}


}//namespace rnn
}//namespace abcdl
