# Problem Session 1 HT23 Parallel Programming Models

+ Name: hongguang Chen
+ CID： chenhon
  
## Problem 1

+ ### questiona a
<!-- why？ -->
Blockwise may achieve the best locality because the cache size is 64 bytes, which means only 16 32-bit numbers can be loaded into the cache. To minimize cache misses, it is best to allocate the entire cache to one processor, reducing cache updates. Let's take the Cyclic type as an example. If P=4, and for some reason, P_4 runs faster than P_1, and P_4 tries to access a[7] while P_1 is trying to access a[0], there will be cache misses for a[7], followed by a cache update, and then another miss for a[0]. This results in more cache misses

### answer
?

+ ### question b
in order to fully use all processor, task size should be P*N, N>=1

granularity = N/(P*M)
M=1


+ ### question c
THE coarsetst grnularity task size should be N/P.
so the execution time should be $ P * 1_{ms}+\frac{N}{P}* 1_{us}$

<!-- # why just 1 ms for schedualing ? -->
P=8
10^-3 * N/P + 1
10^6/8 = 1.25 * 10^5

125 ms + 1 = 126 ms 


+ ### question d
according to question c, when P = 10000, the execution time = $10^4 * + \frac{10^6}{10^4}*10^{-3} ≈ 10^4 ms $ + 

10^-3 * N/P + 1

+ ### question e
if we want schedualing time less than 10%.
we will have：
$$\frac{T_{size}}{P+\frac{P}{N}*10.3} \lt 0.1 $$



+ ### question f
<!-- why dynamic ? -->
G=100,
task = N/G = 10^6 / 10^2 = 10^4
10^4/ P = 10^2 ms
10^2 x 82 ms + 100* 100* 10^-3  =200 + 10 =210 ms

## Problem 2
for shared memory 20ms

a) A(P_0) B(P_0) C(P_0) -> 30ms
b) A(P_0) B(P_1) C(P_0) -> 10 + 15 
<!-- ? why 15 -->
c) A(P_0) B(P_0) C(P_0) -> 30ms
d) A(P_0) B(P_1) C(P_0)
10 + 1000*8/2/10^6 * 2 *10^3 + 15 ->


## Problem 3
### q1 yes
a and b are independent

### q2 yes 
it can be rewritten as:
```
for (i=0;i<N-1l; i++){
  b[i] = a[i+1]
}
for (i=0;i<N-1;i++){
  c[i] = a[i] + a[i+1]
}
```

### q3 yes
?

## Problem 4


## Problem 5
