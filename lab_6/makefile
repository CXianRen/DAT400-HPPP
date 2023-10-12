
GCC=nvcc

BUILD_PATH=build


v1:
	${GCC} -o ./${BUILD_PATH}/matmul_v1 cuda_mm.cu

v2:
	${GCC} -o ./${BUILD_PATH}/matmul_v2_16 cuda_mm_tile.cu -DTILE_SIZE=16
	${GCC} -o ./${BUILD_PATH}/matmul_v2_32 cuda_mm_tile.cu -DTILE_SIZE=32
	${GCC} -o ./${BUILD_PATH}/matmul_v2_64 cuda_mm_tile.cu -DTILE_SIZE=64


run_v1:
	. ./setup.sh; ./${BUILD_PATH}/matmul_v1


run_v2:
	clear
	@echo "----------"
# t = 256 == tile size, n = 32
	./${BUILD_PATH}/matmul_v2_16 32 256
	@echo "----------"
# t = 256 == tile size, n = 64
	./${BUILD_PATH}/matmul_v2_16 64 256
	@echo "----------"
# t = 256 == tile size, n = 128
	./${BUILD_PATH}/matmul_v2_16 128 256
	@echo "----------"
# t = 256 == tile size, n = 128
	./${BUILD_PATH}/matmul_v2_16 160 256
	@echo "----------"
# t = 256 == tile size, n = 128
	./${BUILD_PATH}/matmul_v2_16 192 256
	@echo "----------"
# t = 256 == tile size, n = 256
	./${BUILD_PATH}/matmul_v2_16 256 256
	@echo "----------"
# t = 256 == tile size, n = 256
	./${BUILD_PATH}/matmul_v2_32 256 1024
	@echo "----------"
# t = 256 == tile size, n = 1024
	./${BUILD_PATH}/matmul_v2_16 1024 256
	@echo "----------"
# t = 128 < tile size,  n = 1024
	./${BUILD_PATH}/matmul_v2_16 1024 128
	@echo "----------"
# t = 512 > tile size,  n = 1024
	./${BUILD_PATH}/matmul_v2_16 1024 512
	@echo "----------"
# t = 512 > tile size,  n = 1024
	./${BUILD_PATH}/matmul_v2_32 1024 512
	@echo "----------"
# t = 512 > tile size,  n = 1024
	./${BUILD_PATH}/matmul_v2_64 1024 512
	@echo "----------"
# t = 512 > tile size,  n = 1024
	./${BUILD_PATH}/matmul_v2_16 31  256
	@echo "----------"
# t = 512 > tile size,  n = 1024
	./${BUILD_PATH}/matmul_v2_16 33  256
	@echo "----------"
# t = 512 > tile size,  n = 1024
	./${BUILD_PATH}/matmul_v2_16 40  256
	@echo "----------"
# t = 512 > tile size,  n = 1024
	./${BUILD_PATH}/matmul_v2_16 1023  256
	@echo "----------"
# t = 512 > tile size,  n = 1024
	./${BUILD_PATH}/matmul_v2_16 1025  256

clean:
	rm -rf ./${BUILD_PATH}/*
