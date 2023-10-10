
GCC=nvcc

BUILD_PATH=build


v1:
	${GCC} -o ./${BUILD_PATH}/matmul_v1 cuda_mm.cu

run_v1:
	. ./setup.sh; ./${BUILD_PATH}/matmul_v1

clean:
	rm -rf ./${BUILD_PATH}/*
