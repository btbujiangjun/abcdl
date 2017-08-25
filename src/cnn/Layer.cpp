/*********************************************
* Author: Jun Jiang - jiangjun4@sina.com
* Created: 2017-05-22 16:47
* Last modified: 2017-05-22 16:47
* Filename: CNN.cpp
* Description: convolutional network 
**********************************************/
#include <typeinfo>
#include <math.h>
#include "algorithm/cnn/Layer.h"

namespace abcdl{
namespace cnn{

bool DataLayer::initialize(Layer* pre_layer){
    return true;
}
void DataLayer::feed_forward(Layer* pre_layer, bool debug){
    auto activation = new ccma::algebra::DenseMatrixT<real>();
    _x->clone(activation);
    this->set_activation(0, activation);
    if(debug){
    	printf("DataLayer activation");
        auto a = new ccma::algebra::DenseMatrixT<int>();
        int* d = new int[_x->get_rows() * _x->get_cols()];
		uint size = _x->get_size();

        for(uint i = 0; i != size; i++){
            d[i] = static_cast<int>(_x->get_data(i));
        }

        a->set_shallow_data(d, _x->get_rows(), _x->get_cols());
        a->display("|");
        delete a;
    }
}
void DataLayer::back_propagation(Layer* pre_layer, Layer* back_layer, bool debug){
}

bool SubSamplingLayer::initialize(Layer* pre_layer){
    uint pre_rows = pre_layer->get_rows();
    uint pre_cols = pre_layer->get_cols();

    if(pre_rows % _scale != 0 || pre_cols % _scale != 0){
        printf("SubSampling Layer scale error:pre_rows[%d]pre_cols[%d]scale[%d].\n", pre_rows, pre_cols, _scale);
        return false;
    }
    this->_rows = pre_rows / _scale;
    this->_cols = pre_cols / _scale;
    //can't change out channel size.
    this->_in_channel_size = this->_out_channel_size = pre_layer->get_out_channel_size();
    /*
     * pre_layer, each channel share a bias and initialize value is zero.
     * no pooling weight.
     */
    set_bias(new ccma::algebra::DenseColMatrixT<real>(this->_out_channel_size, 0.0));
    return true;
}
void SubSamplingLayer::feed_forward(Layer* pre_layer, bool debug){
    // in_channel size equal out_channel size to subsampling layer.
    for(uint i = 0; i != this->_in_channel_size; i++){
        auto a = pre_layer->get_activation(i);
        auto activation = new ccma::algebra::DenseMatrixT<real>();
        _pooling->pool(a, this->_rows, this->_cols, this->_scale, activation);
        this->set_activation(i, activation);

	    if(debug){
	        printf("sub feed[%d]", i);
	        activation->display("|");
	    }
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

bool ConvolutionLayer::initialize(Layer* pre_layer){
    uint pre_rows = pre_layer->get_rows();
    uint pre_cols = pre_layer->get_cols();

    if(pre_rows < _kernal_size || pre_cols < _kernal_size){
        printf("ConvolutionLayer Size Erorr: pre_rows[%d] pre_cols[%d] less than kernal_size[%d].\n", pre_rows, pre_cols, _kernal_size);
        return false;
    }

    this->_rows = (pre_rows - _kernal_size) % _stride == 0 ? (pre_rows - _kernal_size) / _stride + 1 : (pre_rows - _kernal_size) / _stride + 2;
    this->_cols = (pre_cols - _kernal_size) % _stride == 0 ? (pre_cols - _kernal_size) / _stride + 1 : (pre_cols - _kernal_size) / _stride + 2;

    this->_in_channel_size = pre_layer->get_out_channel_size();

    for(uint i = 0; i != this->_in_channel_size; i++){
        for(uint j = 0; j != this->_out_channel_size; j++){
            auto weight = new ccma::algebra::DenseRandomMatrixT<real>(_kernal_size, _kernal_size, 0.0, 0.5);
            this->set_weight(i, j, weight);
        }
    }
    for(uint i = 0; i != this->_in_channel_size; i++){
        for(uint j = 0; j != this->_out_channel_size; j++){
            this->get_weight(i, j)->display("|");
        }
    }
    /*
     * channel shared the same bias of current layer.
     */
    this->set_bias(new ccma::algebra::DenseColMatrixT<real>(this->_out_channel_size, 0.0));

    return true;
}

void ConvolutionLayer::feed_forward(Layer* pre_layer, bool debug){
    auto a = new ccma::algebra::DenseMatrixT<real>();
    //foreach output channel
    for(uint i = 0; i != this->_out_channel_size; i++){
        auto activation = new ccma::algebra::DenseMatrixT<real>();
        for(uint j = 0; j != pre_layer->get_out_channel_size(); j++){
            pre_layer->get_activation(j)->clone(a);
            a->convn(this->get_weight(j, i), _stride, "valid");
            //sum all channels of pre_layer.
            activation->add(a);

            if(debug){
                printf("ConvolutionLayer convn[%d][%d]", i, j);
                pre_layer->get_activation(j)->display("|");
                this->get_weight(j, i)->display("|");
                a->display("|");
                activation->display("|");
            }
        }
        //add shared bias of channel in current layer.
        activation->add(this->get_bias()->get_data(i, 0));

        if(debug){
            printf("ConvolutionLayer bias [%d][%f]", i, this->get_bias()->get_data(i, 0));
            activation->display("|");
        }

        //if sigmoid activative function.
        activation->sigmoid();
        this->set_activation(i, activation);
	
	    if(debug){
	        printf("conv feed activation[%d]", i);
	        activation->display("|");
    	}

    }
    delete a;
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

bool FullConnectionLayer::initialize(Layer* pre_layer){
    _cols = pre_layer->get_rows() * pre_layer->get_cols() * pre_layer->get_out_channel_size();
    this->_in_channel_size = pre_layer->get_out_channel_size();

    this->set_bias(new ccma::algebra::DenseColMatrixT<real>(_rows, 0.0));
    auto weight = new ccma::algebra::DenseRandomMatrixT<real>(this->_rows, this->_cols, 0.0, 0.5);
    this->set_weight(0, 0, weight);
    return true;
}
void FullConnectionLayer::feed_forward(Layer* pre_layer, bool debug){
    /*
     * concatenate pre_layer's all channel mat into vector
     */
    auto a = new ccma::algebra::DenseMatrixT<real>();
    auto av = new ccma::algebra::DenseMatrixT<real>();
	uint size = pre_layer->get_rows() * pre_layer->get_cols();;
    for(uint i = 0; i != this->_in_channel_size; i++){
        pre_layer->get_activation(i)->clone(a);
        a->reshape(size, 1);
        av->extend(a, false);
    }
    delete a;
    if(_av != nullptr){
        delete _av;
    }
    _av = av;

    auto activation = new ccma::algebra::DenseMatrixT<real>();
    this->get_weight(0, 0)->clone(activation);
    activation->dot(_av);
    activation->add(this->get_bias());
    //if sigmoid activative function
    activation->sigmoid();
    this->set_activation(0, activation);
    
    if(debug){
        printf("FullConnectionLayer feed_forward activation");
	    activation->display("|");
    }

}

void FullConnectionLayer::back_propagation(Layer* pre_layer, Layer* back_layer, bool debug){
    if(_error == nullptr){
        _error = new ccma::algebra::DenseMatrixT<real>();
    }
    this->get_activation(0)->clone(_error);
    _error->subtract(_y);

	if(debug){
		printf("FullConnectionLayer back activation");
		this->get_activation(0)->display("|");
		printf("FullConnectionLayer back error");
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
        printf("FullConnectionLayer back_propagation derivate_weight");
        derivate_weight->display("|");
        printf("FullConnectionLayer back_propagation derivate_bias");
        derivate_bias->display("|");
    }

    delete derivate_weight;
    delete derivate_bias;
}

}//namespace cnn
}//namespace abcdl
