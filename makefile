CC=g++
all:
#	${CC} -o matrix_test -std=c++11 example/algebra/Matrix.cpp src/algebra/MatrixBase.cpp src/algebra/MatrixOperator.cpp src/algebra/MatrixAlgebra.cpp src/algebra/MatrixHelper.cpp src/utils/Log.cpp -g -pthread -I include/ -O3 -Wall
	${CC} -o dnn_test -std=c++11 example/dnn.cpp src/dnn/Layer.cpp src/dnn/DNN.cpp src/algebra/MatrixBase.cpp src/algebra/MatrixOperator.cpp src/algebra/MatrixAlgebra.cpp src/algebra/MatrixHelper.cpp src/utils/Log.cpp -g -pthread -I include/ -Wall
clean:
	rm -rf matrix_test* &
	rm -rf dnn_test* &
