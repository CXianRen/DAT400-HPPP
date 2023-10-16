# Problem 1
(a) This code can not be paralleized or vectorized. Calculating the c[i] and d[i+1] can be paralliezed and vectorized, becasue these step are independent. But calculating the sum of a[i] * b[i] can not be. In each step, the value of sum relies on its' previous value. Here is the change to the code eanbling it to be paralleized.
```cpp
float sum_arr[N];
float a[N],b[N],c[N], d[N+1];
// main loop
for (auto i = 0; i< N; i++>){
    c[i] = a[i] + b[i];
    sum_arr[i] = a[i] * b[i];
    d[i+1] = d[i];
}
// reduction loop
for (auto i = 0; i< N; i++){
    sum+=sum_arr[N];
}
```
Then the main loop can be parallelized and vectorized. As for the reducting loop, it can be run in parallelized by a binary-reduction method, but still cannot be vectorized.

(b) 
+ Processes:  
advantage: (1) have more cpu time to run (no sharing the time of its' parent process). (2) can run in different cores.  
disadvantage: (1) take much mor time to creat a child process than a thread (slow in creating), (2) resource is not shared among other process,which means have to use other API/libs to communicate. (3) the schedule is controlled by kernel not the user.

+ kernel-level threads:  
advantage: (1) create faster than process, (2) requires less resource (RAM),(3) can run in different cores. (3) can eassily communicate with other threads.  
disadvantage: (1) schedule is controlled by kernel not the user. (2) the actually number of kernel-level threads might be less than the task granularity.

+ user-level threads:  
advantage: (1) create faster than process (2) requires less resource, (3) schedule is controlled by the user. (4) easy to communicate with other threads. 
disadvantage: (1) all threads share same cpu period. (2) only run in a core of the process runing in. 

(c)
+ Process:  
No. When N is less than the number of cores, the overhead of scheduling in process might larger than that of the task. And when N is larger more than the number of cores, for reducing the time of scheduling,  it should keep the number of process same as the cores to avoid wasting to much time in scheduling, so it should not make each iteration be mapped into a distinct process.

+ kernel-level thread:  
No. Same as the condition of process, althought overhead of scheduling in thread level is lower, it also will waste time if there are too many threads are waiting to be executed. So it also depend on the number of N. if N is not that big, it is ok to map each iteration into a single kernel-level thread, but when N is too large, like over the max kernel-level threads the system can provide, the answer is no.

+ user-level thread:  
No. Similar to the condition of kenel-level thread. Although the number of user-level thread is controlled fully by the programmer, but it still required resource like ram, and also all user-level threads share the same cpu period of the process, so when N is coming too larger, mapping each iteration to a distinct threads will leading bad performance.

---
# Problem 2
(a)  
static schedule: 256 chuncks, 1 iteration per chuncks;   
dynamic schedule: 256 chuncks, 1 iteration per chuncks;   
dynamic schedule with chunk size: 6 chuncks, 50 iterations per chunk for the first five chuncks, and 6 iteraitons for the last chunck;  
guide schedule: (TODO?)

(b) when the overhead of scheduling is ignored.  
static schedule: 64 ns  
dynamic schedule: 64 ns  
dynamic with chunk size: 100 ns  
guide schedule: (TODO?)  

(c)  
<!-- self scheduling also call dynamic schdulling? -->
For self-schedulling:  
$T_{exe} = 1ns * 256/4  + 0.1ns * 256/4 = 70.4 ns$

For chunk size = 50:  
$T_{exe} = 50 * 1 ns * 2 + 4.5 ns * 2  = 109 ns$
<!-- is there also overhead when it start ?-->
self-scheduling will run faster.

(d)  
assuming N is the problem size, then we have:  
$$T_{P=32,maxN}= T_{generating}+ T_{runing} < T_{P=4,N=256}$$
=>   
$$N/32 + 1 + 32* 0.5 ns < 64 ns$$
=>  
$$N < (64+17) * 32 = 2592 $$

---
# Problem 3 
<!-- todo  -->
(a)  
there is 6 float point operations, 7 memory accesses (6 read and  2 write); So  
AI =  6 / (8 * 4) = 0.1875 Float/Byte 
<!-- ? should we consider the code optimization? -->

B = 16GB/s, then At = AI * B = 3 GFLOPS  

For a sing core:  
$peak = 4$ GFLOPS  
and    
$T_{exe} = 6 * 10^6 / min(peak, AI * B) = 6* 10^6 / 3GFLOPS \approx 2 ms $


For 8 cores:  
$peak = 4 * 8 = 32$ GFLOPS > 3 GFLOPS,  
so   
$T_{exe} = 6 * 10^6 / min(peak, AI * B) = 6* 10^6 / 3GFLOPS \approx 2 ms $


(b)  
on a single core, $T_{exe} \approx 2ms $, so the time of one iteration of this loop is 2ms + 0.5 ms = 2.5ms, so $f = 2 /2.5 = 0.8$; 
so    
$S_{upper-limmit} = (1)/(1-f+ f/P) = 8/2.4 \approx 3.3 $

<!-- todo? something wrong with AI? -->

For 8 corse:  
$T_{exe} = T_{2dfunc} + T_{serial} = 2.5 ms$
$SpeedUp = T_{8cores} / T_{single} = 1$  

So, (i) executing the kernel on more cores won't improve the speedup, becasue the speed of 2dfunc is limited by the memory bandwidth. (ii) to achieve higher AI can make effect, as AI increase, 2dfunc can utilize the compuation abilibty to its full exent.  (iii)  optimiazing the serial part of the compuation also can improve the performance, but just little, because the serial part accounts for only 20% overhead.


(c)  
