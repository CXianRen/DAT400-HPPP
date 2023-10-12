#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <time.h>

//CUDA RunTime API
#include <cuda_runtime.h>

#define MATRIX_SIZE 1024

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
    // n blocks == n tils, each block handle n/2 lines
    int blk_each_dim = __dsqrt_rn((gridDim.x*gridDim.y* gridDim.z));
    int bid = blockIdx.z * (gridDim.x * gridDim.y) + blockIdx.y* gridDim.x + blockIdx.x;

    int bid_y = bid / blk_each_dim; 
    int bid_x = bid % blk_each_dim;

    int total_thread = blockDim.x * blockDim.y * blockDim.z;
    int tid = threadIdx.z * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x;

    // how many rows in a tile;
    int blk_step = n / blk_each_dim + (n % blk_each_dim>0);
    // how many elements in each tile 
    int elements = blk_step * blk_step;
    // how many elements each thread should calculate.
    int step = elements/total_thread + (elements%total_thread>0);
    
    int temp_tid = 0, ty = 0, tx =0, row = 0, col = 0;
    float sum = 0.0;
    for(int s=0; s<step; s++){
        temp_tid = tid + s * total_thread;
        // if(temp_tid < elements){
            // y in a tile
            ty = temp_tid / blk_step;
            // x in a tile
            tx = temp_tid % blk_step;

            // row in A
            row = bid_y * blk_step + ty;
            // col in A
            col = bid_x * blk_step + tx;

            sum = 0;
            // in case n % blk_each_dim !=0
            if(row<n && col<n){
                for(int i=0;i<n;i++){
                    sum+= a[row*n+i] * b[i*n + col];
                }
                c[row * n + col] = sum;
            }
        // } 
    }
}

int main(int argc, char** argv)
{   
    int n = MATRIX_SIZE;
    int gx=4,gy=1,gz=1;
    int bx=256,by=1,bz=1;

    if(argc < 8){
        printf("use default griddim (4,1,1), blockdim(256,1,1)\n");
    }else
    {
        gx = atoi(argv[1]);
        gy = atoi(argv[2]);
        gz = atoi(argv[3]);

        bx = atoi(argv[4]);
        by = atoi(argv[5]);
        bz = atoi(argv[6]);

        n = atoi(argv[7]);

    }



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
    dim3 dimGrid(gx,gy,gz);
    dim3 dimBlock(bx,by,bz);

    // Kernel Execution

    printf("dimGrid (%d %d %d), dimBlock (%d %d %d), n:%d \n", gx,gy,gz, bx, by, bz, n);
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
