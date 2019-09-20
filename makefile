CC=g++
all:
	${CC} -o libsvm_test -std=c++11 example/algebra/Libsvm.cpp src/algebra/MatrixBase.cpp src/algebra/MatrixOperator.cpp src/algebra/MatrixAlgebra.cpp src/algebra/MatrixHelper.cpp src/utils/Log.cpp -g -pthread -I include/ -O3 -Wall
	${CC} -o matrix_test -std=c++11 example/algebra/Matrix.cpp src/algebra/MatrixBase.cpp src/algebra/MatrixOperator.cpp src/algebra/MatrixAlgebra.cpp src/algebra/MatrixHelper.cpp src/utils/Log.cpp -g -pthread -I include/ -O3 -Wall
	${CC} -o fnn_mnist -std=c++11 example/fnn.cpp src/fnn/Layer.cpp src/fnn/FNN.cpp src/algebra/MatrixBase.cpp src/algebra/MatrixOperator.cpp src/algebra/MatrixAlgebra.cpp src/algebra/MatrixHelper.cpp src/utils/Log.cpp -g -pthread -I include/ -Wall -O3
	${CC} -o sessionq -std=c++11 example/sessionq.cpp src/fnn/Layer.cpp src/fnn/FNN.cpp src/algebra/MatrixBase.cpp src/algebra/MatrixOperator.cpp src/algebra/MatrixAlgebra.cpp src/algebra/MatrixHelper.cpp src/utils/Log.cpp -g -pthread -I include/ -Wall -O3
	${CC} -o cnn_mnist -std=c++11 example/cnn.cpp src/cnn/Layer.cpp src/cnn/CNN.cpp src/framework/Pool.cpp src/algebra/MatrixBase.cpp src/algebra/MatrixOperator.cpp src/algebra/MatrixAlgebra.cpp src/algebra/MatrixHelper.cpp src/utils/Log.cpp -g -pthread -I include/ -Wall -O3
	${CC} -o rnn_test -std=c++11 example/rnn.cpp src/rnn/Layer.cpp src/rnn/RNN.cpp src/algebra/MatrixBase.cpp src/algebra/MatrixOperator.cpp src/algebra/MatrixAlgebra.cpp src/algebra/MatrixHelper.cpp src/utils/Log.cpp -pthread -I include/ -Wall -g -O3 -ggdb
clean:
	rm -rf libsvm_test* &
	rm -rf matrix_test* &
	rm -rf sessionq* &
	rm -rf fnn_mnist* &
	rm -rf cnn_mnist* &
	rm -rf rnn_test* &
