
## Task 1
we measure the start time and end time between Feed forward and Back propagation, and count the value of start - end. the code like this:
```cpp
 // get start time 
    gettimeofday(start,nullptr);
    // Feed forward
    vector<float> a1 = relu(dot( b_X, W1, BATCH_SIZE, 784, 128 ));
    // ...
    // Back propagation
    // ...
    vector<float> dW1 = dot(transform( &b_X[0], BATCH_SIZE, 784 ), dz1, 784, BATCH_SIZE, 128);
    gettimeofday(end,nullptr); 
 
 // print after every 100 epoch loops
    double diff_t = (end->tv_sec-start->tv_sec) + (end->tv_usec-start->tv_usec)/1000.0/1000.0;
    cout << "Forward and Backward Time(s) per epoch:" << 
      diff_t << " " << (diff_t/ticks)*100 <<"% time spend at dot in an epoch" <<endl;
```
and the result like this:
```bash 
Iteration #: 999
Iteration Time: 0.0720264s
Forward and Backward Time(s) per epoch:0.071397 99.1262% time spend at dot in an epoch
Loss: 0.886171
```
And we can see that about 98%-99% time was spent at GEMM kernal.  

## Task 2
the loop interchange code:
``` cpp
    // loop interchange 
    for( int row = 0; row < m1_rows; ++row ) {
        for( int k = 0; k < m1_columns; ++k ){
            for( int col = 0; col < m2_columns; ++col ) {
                output[ row * m2_columns + col ] += m1[ row * m1_columns + k ] * m2[ k * m2_columns + col ];
            }
        }
    }
```
and the result like this:
```bash
Iteration #: 999
Iteration Time: 0.0122538s
Forward and Backward Time(s) per epoch:0.011546 94.2238% time spend at dot in an epoch
Loss: 0.886171
```
it is obviously, time spent at dot reduce many, about 1 out of 5 to 6 of orignial version.

## Task 3
we do tile for i,j,k speratively, then we get the nested loops like this:
```cpp
#define BLOCK_TILE 
//...
for(int it=0;it<N; it+=block_size)
   for(int kt=0;kt<K;kt+=block_size)
      for(int jt=0;jt<M;jt+=block_size)
            for(int i=it;i<min(it+block_size,N);i++)
               for(int k=kt; k<min(kt+block_size,K); k++)
                  for(int j=jt;j<min(jt+block_size,M);j++)
                        output[i*M+j] +=m1[i*K+k]* m2[k*M+j];
//..
```
and we get time consumption:
```bash
Iteration #: 999
Iteration Time: 0.033847s
Forward and Backward Time(s) per epoch:0.032891 97.1755% time spend at dot in an epoch
Loss: 0.886171
```
# TODO: what would be an efficient BLOCK_SIZE?


## Task 4
### Step a and b
we acheive mutlithread by this code:
``` cpp
    const int num_partitions = 4; // the number of processor ?
    pthread_t threads[num_partitions];
    const int step = m1_rows / num_partitions + (m1_rows % num_partitions != 0);
    for (int i = 0; i < num_partitions; ++i) {
      gemm_thread_args* args = new gemm_thread_args;
      args->output = &output;
      // assign rest of the arguments of gemm_thread_args accordingly
      args->m1= &m1;
      args->m2= &m2;
      args->m1_rows = m1_rows;
      args->m1_columns = m1_columns;
      args->m2_columns = m2_columns;
      args->row_start = step*i;
      args->row_end = min(args->row_start+ step, m1_rows);
      pthread_create(&threads[i],nullptr,[](void* args)->void*{
        gemm_thread_args* margs = (gemm_thread_args*)args;
        for( int row = margs->row_start; row < margs->row_end; ++row ) {
            for( int k = 0; k < margs->m1_columns; ++k ){
                for( int col = 0; col < margs->m2_columns; ++col ) {
                    (*(margs->output))[ row * margs->m2_columns + col ] += 
                        (*(margs->m1))[ row * margs->m1_columns + k] *
                        (*(margs->m2))[ k * margs->m2_columns + col ];
                }
            }
        }
        return NULL;
      }, args); 
    }
    for (int i = 0; i < num_partitions; ++i) {
      pthread_join(threads[i], nullptr);
    }
```
then run to collect time consumption:
```bash
Iteration #: 999
Iteration Time: 0.0165817s
Forward and Backward Time(s) per epoch:0.015506 93.513% time spend at dot in an epoch
Loss: 0.886171
```
we have tested for different thread number, from 2 to 12.
<!-- todo plot it -->
### Step c
todo

## Task 5
### Step a, the maximum speedup 