CC=g++
all:
	${CC} -o algebra.matrix -std=c++11 example/algebra/Matrix.cpp src/algebra/MatrixBase.cpp src/algebra/MatrixOperator.cpp src/algebra/MatrixAlgebra.cpp src/algebra/MatrixHelper.cpp -g -pthread -I ./include/ -O3 -Wall
clean:
	rm -f algebra.matrix* &