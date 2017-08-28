/*********************************************
* Author: Jun Jiang - jiangjun4@sina.com
* Created: 2017-05-22 16:47
* Last modified: 2017-05-22 16:47
* Filename: CNN.cpp
* Description: convolutional network 
**********************************************/
#include <math.h>
#include "cnn/Layer.h"

namespace abcdl{
namespace cnn{

void SubSamplingLayer::initialize(Layer* pre_layer){
    uint pre_rows = pre_layer.rows();
    uint pre_cols = pre_layer.cols();
    CHECK(pre_rows % _scale == 0 && pre_cols % _scale == 0);

	
    this->_rows = pre_rows / _scale;
    this->_cols = pre_cols / _scale;

    //can't change out_channel_size.
    this->_in_channel_size = this->_out_channel_size = pre_layer->get_out_channel_size();

    /*
     * pre_layer, all channels share the same bias and initial value is zero.
     * no pooling weight.
     */
    this->_bias.reset(0.0, this->_out_channel_size, 1);
}
void SubSamplingLayer::forward(Layer* pre_layer){
    // in_channel size equal out_channel size to subsampling layer.
	abcdl::algebra::Mat activation;
    for(size_t i = 0; i != this->_in_channel_size; i++){
        _pooling->pool(activation, pre_layer->get_activation(i), this->_rows, this->_cols, this->_scale);
        this->set_activation(activation, i);
    }
}
void SubSamplingLayer::back_propagation(Layer* pre_layer, Layer* back_layer, bool debug){
    if(back_layer->get_is_last_layer()){
        real* data = back_layer->get_delta(0)->get_data();
        for(uint i = 0; i != this->_out_channel_size; i++){
            auto delta = new ccma::algebra::DenseMatrixT<real>();
            real* d = new real[_rows * _cols];
            memcpy(d, &data[i * this->_rows * this->_cols], sizeof(real) * this->_rows * this->_cols);
            delta->set_shallow_data(d, this->_rows, this->_cols);
            this->set_delta(i, delta);

            if(debug){
	        	printf("sub back-full[%d]", i);
	            delta->display("|");
            }
        }
    }else if(typeid(*back_layer) == typeid(ConvolutionLayer)){
        uint stride = ((ConvolutionLayer*)back_layer)->get_stride();
        auto d = new ccma::algebra::DenseMatrixT<real>();
        auto w = new ccma::algebra::DenseMatrixT<real>();
        for(uint i = 0 ; i != this->_out_channel_size; i++){
            auto delta = new ccma::algebra::DenseMatrixT<real>();
            for(uint j = 0; j != back_layer->get_out_channel_size(); j++){
                back_layer->get_delta(j)->clone(d);
                back_layer->get_weight(i, j)->clone(w);
                d->convn(w, stride, "full");
                delta->add(d);
            }
            this->set_delta(i, delta);

            if(debug){
	        	printf("sub back-conv[%d]", i);
	        	delta->display("|");
            }
	
        }
        delete d;
        delete w;
    }
}

void ConvolutionLayer::initialize(Layer* pre_layer){
    uint pre_rows = pre_layer->get_rows();
    uint pre_cols = pre_layer->get_cols();

    CHECK(pre_rows > _kernal_size && pre_cols < _kernal_size);

    this->_rows = (pre_rows - _kernal_size) % _stride == 0 ? (pre_rows - _kernal_size) / _stride + 1 : (pre_rows - _kernal_size) / _stride + 2;
    this->_cols = (pre_cols - _kernal_size) % _stride == 0 ? (pre_cols - _kernal_size) / _stride + 1 : (pre_cols - _kernal_size) / _stride + 2;

    this->_in_channel_size = pre_layer->get_out_channel_size();
	
	this->_weights.reserve(this->_in_channel_size * this->_out_channel_size);
	for(auto& weight : _weights){
		weight.reset(_kernal_size, _kernal_size, 0.0, 0.5);
	}

	for(auto& weight : _weight){
		weight.display("|");
	}

    /*
     * all channels shared the same bias of current layer.
     */
	this->_bias.reset(0.0, this->_out_channel_size, 1);
}

void ConvolutionLayer::forward(Layer* pre_layer){
    for(size_t i = 0; i != this->_out_channel_size; i++){
        abcdl::algebra::Mat activation;
        for(size_t j = 0; j != pre_layer->get_out_channel_size(); j++){
            auto pre_activation = pre_layer->get_activation(j);
            pre_activation.convn(this->get_weight(j, i), _stride, abcdl::algebra::VALID);
            activation += pre_activation;//sum all channels of pre_layer
        }
        //add shared bias of channel in current layer.
        activation += this->_bias().get_data(i, 0);

        //if sigmoid activative function.
        activation.sigmoid();
        this->set_activation(activation, i);
    }
}

void ConvolutionLayer::back_propagation(Layer* pre_layer, Layer* back_layer, bool debug){
    if(back_layer->get_is_last_layer()){
        real* data = back_layer->get_delta(0)->get_data();
		uint size = this->_rows * this->_cols;
        for(uint i = 0; i != this->_out_channel_size; i++){
            auto delta = new ccma::algebra::DenseMatrixT<real>();
            real* d = new real[size];
            memcpy(d, &data[i * size], sizeof(real) * size);
            delta->set_shallow_data(d, this->_rows, this->_cols);
            this->set_delta(i, delta);

            if(debug){
                printf("ConvolutionLayer-last_layer back_propagation[%d]\n", i);
                delta->display("|");
            }
        }
    }else if(typeid(*back_layer) == typeid(SubSamplingLayer)){

        SubSamplingLayer* sub_layer = (SubSamplingLayer*)back_layer;
        uint scale = sub_layer->get_scale();

        for(uint i = 0; i != this->_out_channel_size; i++){
            /*
             * derivative_sigmoid: 
             *  sigmoid(z)*(1-sigmoid(z))
             *  a = sigmoid(z)
             */
            auto delta = new ccma::algebra::DenseMatrixT<real>();
            auto d = new ccma::algebra::DenseMatrixT<real>();
            this->get_activation(i)->clone(delta);
            delta->clone(d);
			
    	    d->multiply(-1);
            d->add(1);
            delta->multiply(d);
            delete d;

            auto back_delta = new ccma::algebra::DenseMatrixT<real>();
            back_layer->get_delta(i)->clone(back_delta);

            /*
             * subsampling layer reduced matrix dim, so recover it by expand function
             */
            back_delta->expand(scale, scale);
            /*
             * back layer error sharing
             */
            back_delta->division(scale*scale);
            /*
             * delta_l = derivative_sigmoid * delta_l+1(recover dim)
             */
            delta->multiply(back_delta);
            delete back_delta;

            this->set_delta(i, delta);

            if(debug){
                printf("ConvolutionLayer-none_last_layer back_propagation[%d]\n", i);
                delta->display("|");
            }
	    }
    }
    /*
     * calc grad and update weight/bias
     * only for online learning, if batch learning
     * need to average weight and bias
     */
    auto derivate_weight = new ccma::algebra::DenseMatrixT<real>();
    real* derivate_bias_data = new real[this->_out_channel_size];
    for(uint i = 0; i != this->_out_channel_size; i++){
    	for(uint j = 0; j != pre_layer->get_out_channel_size(); j++){
			//derivate_weight = pre_layer.activation(j).convn(delta[i], 'valid')
	        pre_layer->get_activation(j)->clone(derivate_weight);

            if(debug){
    			derivate_weight->display("|");
	    		this->get_delta(i)->display("|");
            }
		    derivate_weight->convn(this->get_delta(i), _stride, "valid");
    	    /*
             * update grad: w -= alpha * derivate_weight
	         */
            if(debug){
			    printf("convolutelayer old derivate_weight");
			    derivate_weight->display("|");
            }

	        derivate_weight->multiply(this->_alpha);
            
            if(debug){
    			derivate_weight->display("|");
                this->get_weight(j, i)->display("|");
            }

            this->get_weight(j, i)->subtract(derivate_weight);
            
            if(debug){
                this->get_weight(j, i)->display("|");
	            printf("conv back derivate_weight[%d][%d]", j , i);
	            derivate_weight->display("|");
            }
    	}
	    //update bias
        derivate_bias_data[i] = this->get_delta(i)->sum();

		if(debug){
			printf("ConvolutionLayer delta[%d]", i);
			this->get_delta(i)->display("|");
		}
    }
    delete derivate_weight;

    auto derivate_bias = new ccma::algebra::DenseMatrixT<real>();
    derivate_bias->set_shallow_data(derivate_bias_data, this->_out_channel_size, 1);
    derivate_bias->multiply(this->_alpha);
    this->get_bias()->subtract(derivate_bias);
  
    if(debug){
	    printf("conv back derivate_bias");
        derivate_bias->display("|");
    }

    delete derivate_bias;
}

void OutputLayer::initialize(Layer* pre_layer){
	//concat all vector to a array 
    this->_cols = pre_layer.rows() * pre_layer.cols() * pre_layer.get_out_channel_size();
    this->_in_channel_size = pre_layer->get_out_channel_size();

    this->_bias.reset(0.0, _rows, 1);
	ccma::algebra::RandomMatrix<real> weight(this->_rows, this->_cols, 0.0, 0.5);
    this->set_weight(0, 0, weight);
}
void OutputLayer::forward(Layer* pre_layer){
    /*
     * concatenate pre_layer's all channel mat into vector
     */
	size_t size = pre_layer->get_rows() * pre_layer->get_cols();
    T* data = new T[size];
    size_t idx = 0;
    for(size_t i = 0; i != this->_in_channel_size; i++){
        memcpy(data, pre_layer.get_activation(i).data(), sizeof(T) * pre_layer.get_activation(i).get_size());
        idx += pre_layer.get_activation(i).get_size();
    }

    abcdl::algebra::Mat mat;
    mat.set_shallow_data(data, size, 1);

    MatrixHelper<T> mh;
    auto activation = mh.dot(this->get_weight(0, 0), mat) + this->_bias;

    //if sigmoid activative function
    activation->sigmoid();
    this->set_activation(activation, 0);
}

void OutputLayer::back_propagation(Layer* pre_layer, Layer* back_layer, bool debug){
    if(_error == nullptr){
        _error = new ccma::algebra::DenseMatrixT<real>();
    }
    this->get_activation(0)->clone(_error);
    _error->subtract(_y);

	if(debug){
		printf("OutputLayer back activation");
		this->get_activation(0)->display("|");
		printf("OutputLayer back error");
		_error->display("|");
	}

    /* loss function, mse mean square error
     * 1/2 sum(error*error)/size
     * the size is 1 right here
     */
    auto mse_mat = new ccma::algebra::DenseMatrixT<real>();
    _error->clone(mse_mat);
    mse_mat->multiply(mse_mat);
    _loss = mse_mat->sum()/2;
    delete mse_mat;

    /*
     * error * derivate_of_output
     * derivate_of_output is activation * (1-activation)
     */
    auto derivate_output	= new ccma::algebra::DenseMatrixT<real>();
    auto derivate_output_b 	= new ccma::algebra::DenseMatrixT<real>();

    this->get_activation(0)->clone(derivate_output);
    derivate_output->clone(derivate_output_b);
    
	derivate_output_b->multiply(-1);
    derivate_output_b->add(1);

    derivate_output->multiply(derivate_output_b);
    derivate_output->multiply(_error);

    delete derivate_output_b;

    /*
     * calc delta: weight.T * derivate_output
     */
    auto delta = new ccma::algebra::DenseMatrixT<real>();
    this->get_weight(0, 0)->clone(delta);
    delta->transpose();
    delta->dot(derivate_output);
    this->set_delta(0, delta);
    //if pre_layer is ConvolutionLayer, has sigmoid function
    if(typeid(*pre_layer) == typeid(ConvolutionLayer)){
        auto av1 = new ccma::algebra::DenseMatrixT<real>();
        auto av2 = new ccma::algebra::DenseMatrixT<real>();
        _av->clone(av1);
        _av->clone(av2);
        /*
         * derivate_sigmoid: z * (1-z)
         */
        av2->multiply(-1);
        av2->add(1);
        av1->multiply(av2);
        this->get_delta(0)->multiply(av1);

        delete av1;
        delete av2;
    }

    /*
     * derivate_weight = derivate_output * av.T
     * derivate_bias = derviate_output
     */
    auto derivate_weight = new ccma::algebra::DenseMatrixT<real>();
    auto derivate_bias   = new ccma::algebra::DenseMatrixT<real>();
    auto avt             = new ccma::algebra::DenseMatrixT<real>();

    derivate_output->clone(derivate_weight);
    derivate_output->clone(derivate_bias);
    delete derivate_output;
    _av->clone(avt);

    avt->transpose();
    derivate_weight->dot(avt);
    delete avt;

    /*
     * update weight & bias
     */
    //todo set alpha

    if(debug){
    	printf("derivate_weight");
	    derivate_weight->display("|");
    }

	derivate_weight->multiply(this->_alpha);
	
    if(debug){
    	derivate_weight->display("|");
    }

	derivate_bias->multiply(this->_alpha);
   
    if(debug){
    	printf("old weight");
	    this->get_weight(0, 0)->display("|");
    }

	this->get_weight(0, 0)->subtract(derivate_weight);

    if(debug){
	    this->get_weight(0, 0)->display("|");
    }

    this->get_bias()->subtract(derivate_bias);

    if(debug){
        printf("OutputLayer back_propagation derivate_weight");
        derivate_weight->display("|");
        printf("OutputLayer back_propagation derivate_bias");
        derivate_bias->display("|");
    }

    delete derivate_weight;
    delete derivate_bias;
}

}//namespace cnn
}//namespace abcdl
