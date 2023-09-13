
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
Iteration Time: 0.0762976s
Forward and Backward Time(s) per epoch:0.075361 98.7724% time spend at dot in an epoch
Loss: 0.886171
```
And we can see that about 98%-99% time was spent at GEMM kernal.

## Task 2
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
```bash
Iteration #: 999
Iteration Time: 0.0262516s
Forward and Backward Time(s) per epoch:0.025327 96.478% time spend at dot in an epoch
Loss: 0.886171
```
perf result:
```

```

## Task 3
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
```bash
Iteration #: 999
Iteration Time: 0.033847s
Forward and Backward Time(s) per epoch:0.032891 97.1755% time spend at dot in an epoch
Loss: 0.886171
```

## Task 4
threads=10
```bash
Iteration #: 999
Iteration Time: 0.0165817s
Forward and Backward Time(s) per epoch:0.015506 93.513% time spend at dot in an epoch
Loss: 0.886171

```