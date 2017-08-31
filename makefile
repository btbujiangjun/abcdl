CC=g++
all:
	${CC} -o matrix_test -std=c++11 example/algebra/Matrix.cpp src/algebra/MatrixBase.cpp src/algebra/MatrixOperator.cpp src/algebra/MatrixAlgebra.cpp src/algebra/MatrixHelper.cpp src/utils/Log.cpp -g -pthread -I include/ -O3 -Wall
	${CC} -o dnn_test -std=c++11 example/dnn.cpp src/dnn/Layer.cpp src/dnn/DNN.cpp src/algebra/MatrixBase.cpp src/algebra/MatrixOperator.cpp src/algebra/MatrixAlgebra.cpp src/algebra/MatrixHelper.cpp src/utils/Log.cpp -g -pthread -I include/ -Wall -O3
	${CC} -o cnn_test -std=c++11 example/cnn.cpp src/cnn/Layer.cpp src/cnn/CNN.cpp src/framework/Pool.cpp src/algebra/MatrixBase.cpp src/algebra/MatrixOperator.cpp src/algebra/MatrixAlgebra.cpp src/algebra/MatrixHelper.cpp src/utils/Log.cpp -g -pthread -I include/ -Wall
clean:
	rm -rf matrix_test* &
	rm -rf dnn_test* &
	rm -rf cnn_test* &
