#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <time.h>

//CUDA RunTime API
#include <cuda_runtime.h>

#define MATRIX_SIZE 64
#define TILE_SIZE 16

void printDeviceProp(const cudaDeviceProp &prop)
{
    printf("Device Name : %s.\n", prop.name);
    printf("totalGlobalMem : %d.\n", prop.totalGlobalMem);
    printf("sharedMemPerBlock : %d.\n", prop.sharedMemPerBlock);
    printf("regsPerBlock : %d.\n", prop.regsPerBlock);
    printf("warpSize : %d.\n", prop.warpSize);
    printf("memPitch : %d.\n", prop.memPitch);
    printf("maxThreadsPerBlock : %d.\n", prop.maxThreadsPerBlock);
    printf("maxThreadsDim[0 - 2] : %d %d %d.\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("maxGridSize[0 - 2] : %d %d %d.\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("totalConstMem : %d.\n", prop.totalConstMem);
    printf("major.minor : %d.%d.\n", prop.major, prop.minor);
    printf("clockRate : %d.\n", prop.clockRate);
    printf("textureAlignment : %d.\n", prop.textureAlignment);
    printf("deviceOverlap : %d.\n", prop.deviceOverlap);
    printf("multiProcessorCount : %d.\n", prop.multiProcessorCount);
}

//CUDA Initialization
bool InitCUDA()
{
    int count;
    cudaGetDeviceCount(&count);
    if (count == 0) 
    {
        fprintf(stderr, "There is no device.\n");
        return false;
    }

    int i;
    for (i = 0; i < count; i++) 
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printDeviceProp(prop);
        if (cudaGetDeviceProperties(&prop, i) == cudaSuccess) 
        {
            if (prop.major >= 1) 
            {
            break;
            }
        }
    }
    if (i == count) 
    {
        fprintf(stderr, "There is no device supporting CUDA 1.x.\n");
        return false;
    }
    cudaSetDevice(i);
    return true;
}

// Generate Random Matrix Elements
void matgen(float* a, int n)
{
    int i, j;
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            a[i * n + j] = (float)rand() / RAND_MAX + (float)rand() / RAND_MAX / RAND_MAX;
        }
    }
    // for(int i = 1; i<= n*n; i++)
    //     a[i-1] = i;
}

/* Task 1 & 2: Implement Your Kernel Function Here */
__global__ static void matMultCUDA(const float* a, const float* b, float* c, int n)
{
    // n blocks == n tils, each block handle n/2 lines
    int blk_each_dim = __dsqrt_rn((gridDim.x*gridDim.y* gridDim.z));

    int bid = blockIdx.z * (gridDim.x * gridDim.y) + blockIdx.y* gridDim.x + blockIdx.x;

    int bid_y = bid / blk_each_dim; 
    int bid_x = bid % blk_each_dim;

    int total_thread = blockDim.x * blockDim.y * blockDim.z;
    int tid = threadIdx.z * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x;

    // how many rows in a tile;
    int tile_step =TILE_SIZE;
    // how many elements in each tile 
    int elements = tile_step * tile_step;
    // how many elements each thread should calculate.
    int thread_step = elements/total_thread + (elements%total_thread>0);
    
    int temp_tid = 0, ty = 0, tx =0, row = 0, col = 0;
    float sum = 0.0;

    __shared__ float a_tile[TILE_SIZE*TILE_SIZE];
    __shared__ float b_tile[TILE_SIZE*TILE_SIZE];
   
    #define MAT(a,y,x) a[(y)*n+(x)]
    #define MATB(a,by,bx,y,x) a[(y + (by) * TILE_SIZE )*n+(x + (bx) * TILE_SIZE)]
    #define MAT_A_TILE(y,x) a_tile[(y)* TILE_SIZE + (x)]
    #define MAT_B_TILE(y,x) b_tile[(y)* TILE_SIZE + (x)]
    
     // for each tile of C (each block), there blk_each_dim of A * B
    for(int tile_idx = 0; tile_idx < blk_each_dim; tile_idx ++){
        // for each thread, it copy the element a and b into shared memeory
        // which element should be copied in a tile
            ty  = tid / tile_step;
            tx  = tid % tile_step;
            // which element c should be updated
            row = bid_y * tile_step + ty;
            col = bid_x * tile_step + tx;

            // copy tile_idx th tile of A and B
           
            MAT_A_TILE(ty,tx) = MATB(a,bid_y, tile_idx, ty, tx);
            MAT_B_TILE(ty,tx) = MATB(b,tile_idx, bid_x, ty, tx);
                
            if(tile_idx == 0){
                    MAT(c,row, col) = 0;
            }

       
        // wait all threads done the copy process.
        __syncthreads();

        // calculate 
        // if(row < n && col < n){
            sum = 0;
            for( int i = 0; i< tile_step; i++){
                sum +=  MAT_A_TILE(ty,i) * MAT_B_TILE(i,tx);
            }
            MAT(c,row,col) += sum;
        // }

    }

}

int main(int argc, char** argv)
{   
    int n = MATRIX_SIZE;
  
    if (!InitCUDA()) return 0; 

    float *a, *b, *c, *d;



    a = (float*)malloc(sizeof(float)* n * n); 
    b = (float*)malloc(sizeof(float)* n * n); 
    c = (float*)malloc(sizeof(float)* n * n); 
    d = (float*)malloc(sizeof(float)* n * n);

    srand(0);

    matgen(a, n);
    matgen(b, n);

    float *cuda_a, *cuda_b, *cuda_c;

    /* Task: Memory Allocation */
    cudaMalloc(&cuda_a,sizeof(float)* n * n);
    cudaMalloc(&cuda_b,sizeof(float)* n * n);
    cudaMalloc(&cuda_c,sizeof(float)* n * n);
    
    /* Task: CUDA Memory Copy from Host to Device */
    cudaMemcpy(cuda_a,a,sizeof(float)*n*n, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_b,b,sizeof(float)*n*n, cudaMemcpyHostToDevice);

    /* Task: Number of Blocks and Threads && Dimention*/
    int block_dim = n/TILE_SIZE + (n%TILE_SIZE>0);
    dim3 dimGrid(block_dim,block_dim,1);
    dim3 dimBlock(16*16,1,1);

    // Kernel Execution
    printf("dimGrid (%d %d %d), dimBlock (%d %d %d) \n", block_dim,block_dim,1, 256, 1, 1);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    matMultCUDA << < dimGrid, dimBlock >> >(cuda_a , cuda_b , cuda_c , n);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("GPU time: %f\n", milliseconds);

    /* Task: CUDA Memory Copy from Device to Host */
    cudaMemcpy(c,cuda_c, sizeof(float)*n*n, cudaMemcpyDeviceToHost);



    //Free
    cudaFree(cuda_a);
    cudaFree(cuda_b);
    cudaFree(cuda_c);

    clock_t cstart,cend;

    cstart = clock();
    // CPU Implementation of MatMul
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        { 
            double t = 0;
            for (int k = 0; k < n; k++)
            { 
                t += a[i * n + k] * b[k * n + j]; 
            } 
            d[i * n + j] = t; 

        } 
    }
    cend = clock();
    milliseconds = ((double)(cend-cstart))/CLOCKS_PER_SEC * 1000;
   
    printf("CPU time: %f\n", milliseconds);
    // Check the accuracy of GPU results with CPU results
    float max_err = 0;
    float average_err = 0; 
    int first_diff = -1;
    for (int i = 0; i < n; i++) 
    {
        for (int j = 0; j < n; j++) 
        {
            if (d[i * n + j] != 0)
            { 
                float err = fabs((c[i * n + j] - d[i * n + j]) / d[i * n + j]);
                if (max_err < err){
                    max_err = err; 
                    if(err>0){
                        first_diff = i * n + j;
                    }
                } 
                average_err += err; 
             
            } 
        } 
    }
    printf("Max error: %g Average error: %g\n",max_err, average_err / (n * n));
    if(first_diff>=0)
        printf("max different idx: %d,  c[] is %f, d[] is %f\n", first_diff, c[first_diff], d[first_diff]);

    // for(int i = 0; i < MATRIX_SIZE;i ++){
    //     for(int j = 0; j< MATRIX_SIZE; j++)
    //         printf("%f ", c[i*MATRIX_SIZE + j]);
    //     printf("\n");
    // }
    // printf("\n d:\n");

    // for(int i = 0; i < MATRIX_SIZE;i ++){
    //     for(int j = 0; j< MATRIX_SIZE; j++)
    //         printf("%f ", d[i*MATRIX_SIZE + j]);
    //     printf("\n");
    // }
    // printf("\n");
    return 0;
}
