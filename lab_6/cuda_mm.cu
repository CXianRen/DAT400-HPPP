#include <stdio.h>
#include <stdlib.h>
#include <time.h>

//CUDA RunTime API
#include <cuda_runtime.h>

#define MATRIX_SIZE 32

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
}

/* Task 1 & 2: Implement Your Kernel Function Here */
__global__ static void matMultCUDA(const float* a, const float* b, float* c, int n)
{
    // 4 blocks, each block handle n/2 lines
    int blk_each_dim = __dsqrt_rn((gridDim.x*gridDim.y* gridDim.z));
    // int blk_each_dim = 1;

    int bid = blockIdx.z * (gridDim.x * gridDim.y) + blockIdx.y* gridDim.x + blockIdx.x;
    // int bid = blockIdx.x;

    int bid_y = bid / blk_each_dim; 
    int bid_x = bid % blk_each_dim;

    int tid = threadIdx.z * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x;
    // int tid = threadIdx.x;

    int step = n / blk_each_dim;
    // int step = n;

    int ty = tid / step ;
    int tx = tid % step;
    
    //  bid_y th blk in row direction, each blk has step row, each row has n elements,
    int row = bid_y * step + ty;
    // int row = ty;
    //  bid_x th blk in col direction 
    int col = bid_x * step + tx;
    // int col = tx;
    
    float sum = 0;
    if(row<n){
        for(int i=0;i<n;i++){
            sum+= a[row*n+i] * b[i*n + col];
        }
        c[row * n + col] = sum;
    }
}

int main()
{   
    int gx=4,gy=1,gz=1;
    int bx=256,by=1,bz=1;

    if (!InitCUDA()) return 0; 

    float *a, *b, *c, *d;

    int n = MATRIX_SIZE;

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
    dim3 dimGrid(gx,gy,gz);
    dim3 dimBlock(bx,by,bz);

    // Kernel Execution
    matMultCUDA << < dimGrid, dimBlock >> >(cuda_a , cuda_b , cuda_c , n);

    /* Task: CUDA Memory Copy from Device to Host */
    cudaMemcpy(c,cuda_c, sizeof(float)*n*n, cudaMemcpyDeviceToHost);
    
    //Free
    cudaFree(cuda_a);
    cudaFree(cuda_b);
    cudaFree(cuda_c);

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

    // for(int i =32; i< 2* 32;i ++){
    //     printf("%f ", c[i]);
    // }
    // printf("\n");

    // for(int i =32; i< 2* 32;i ++){
    //     printf("%f ", d[i]);
    // }
    // printf("\n");
    return 0;
}
