CC=g++
all:
	${CC} -o sessionq -std=c++11 example/sessionq.cpp src/fnn/Layer.cpp src/fnn/FNN.cpp src/algebra/MatrixBase.cpp src/algebra/MatrixOperator.cpp src/algebra/MatrixAlgebra.cpp src/algebra/MatrixHelper.cpp src/utils/Log.cpp -g -pthread -I include/ -Wall -O3
clean:
	rm -rf libsvm_test* &
	rm -rf matrix_test* &
	rm -rf sessionq* &
	rm -rf fnn_mnist* &
	rm -rf cnn_mnist* &
	rm -rf rnn_test* &
