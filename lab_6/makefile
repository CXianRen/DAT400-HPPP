
GCC=nvcc

BUILD_PATH=build


v1:
	${GCC} -o ./${BUILD_PATH}/matmul_v1 cuda_mm.cu

v2:
	${GCC} -o ./${BUILD_PATH}/matmul_v2 cuda_mm_tile.cu


run_v1:
	. ./setup.sh; ./${BUILD_PATH}/matmul_v1


run_v2:
	. ./setup.sh; ./${BUILD_PATH}/matmul_v2

clean:
	rm -rf ./${BUILD_PATH}/*
