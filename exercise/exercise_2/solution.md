# Session 2 Performance Analysis Problems
+ Name: Hongguang Chen
+ CID: chenhon
## Problem 1
+ Subsection a:
>Processor number $P=8$  
Total sequential execution time  $ T^* = 1000$
Execution time $T_{exe}$:
$$T_{exe} = T_p = T_{comp.} + T_{synchro.} + T_{exchange} + T_{wait}$$
One core local computations time $T_{comp.}$:
$$T_{comp.} = \frac{T^*}{P}$$
Then:
$$T_{exe} = \frac{1000}{8} + 2 + 10  = 125 + 2 + 10 = 137 s$$
Cost $C_p$:
$$C_p = P*T_p = 8*137= 1096 s$$
Efficiency $E$：
$$E = \frac{T^*}{C_p} = \frac{T^*} { P * T_p } = \frac{1000}{1096} = 0.91$$

+ Subsection b
>According part a the exection time $T_{exe}$: 
$$T_{exe} = T_p = max(180, (1000-180)/7) + 2 + 10  =  180 + 2 + 10 = 192s$$
And the Cost $C_p$:
$$C_p = T_p* P = 192 * 8 = 1536s$$
For the efficiency $E$:
$$E = \frac{T^*}{C_p} = \frac{1000}{1536} = 0.65$$


## Problem 2
The iteraion $N=1,000,000$ and the processor number $P=8$, excution time of each iteration $B=10ns$, and for the dynamic scheduling the $\sigma = 200ns$
+ Subsection a
> For **static scheduler**, the excution time $T_{exe}$：
$$T_{exe} = \frac{N}{P} * B  = \frac{10^6}{8} * (10* 10^{-3})us  = 1250 us$$
For **self-scheduling scheduler**, the excution time $T_{exe}$：
$$T_{exe} = \frac{N}{P}*(B+\sigma) = \frac{10^6}{8} * ((10 + 200)*10^-3)us  = 26250 us $$ 
For **chunk scheduler**,the chunk size $k=1000$, then the excution time $T_{exe}$：
$$
\begin{align*}
T_{exe} &= \frac{N}{k*P}*(k*B + \sigma)  \\
&= \frac{10^6}{1000*8}*((1000*10 + 200)*10^-3)us = 1275us
\end{align*}
$$

+ Subsection b
> When overhead of creating thread, $T_{create} = 50us+P*5us $,was taken into account:
For $P=16$:  
**Static scheduler**:
$$T_{exe} = \frac{N}{P} * B + T_{create} = \frac{10^6}{16} * (10* 10^{-3})us + (50us + 16*5us)  = 755 us$$
**Self-scheduling scheduler**:
$$
\begin{align*}
T_{exe} &= \frac{N}{P}*(B+\sigma) + T_{create} \\
        &= \frac{10^6}{16} * ((10 + 200)*10^{-3})us + (50us+16*5us)  \\
        &= 13255 us 
\end{align*}
$$  
**Chunk scheduler**:
$$
\begin{align*}
T_{exe} &= \frac{N}{k*P}*(k*B + \sigma) + T_{create} \\
        &= \frac{10^6}{1000*16}*((1000*10 + 200)*10^{-3})us + {50us+16*5us} \\
        &= 7630 us  
\end{align*}
$$
When $P=32$:
$$
\begin{align*}
  T_{static} &= 482.5 us\\
  T_{self-scheduling} &= 6732.5 us\\
  T_{chunk} &= 3920 us\\
\end{align*}
$$
When $P=64$,the calculation method is the same as when P=16,32. 

+ Subsection c
> Becasue:
$$
\begin{align*}
T_p &= \frac{N}{P} * B + T_{create} \\ 
    &= \frac{10^6}{P}*(10*10^{-3})us + (50us+P*5us) \\
    &= \frac{10^4}{P} + 50us + P*5us
\end{align*}
$$
Differentiate $T_p$ with respect to $P$:
$$
\begin{align*}
  \frac{dT_p}{dP} = \frac{-10^4}{P^2}+5 &= 0 \\
                                      5 &= \frac{10^4}{P^2} \\
                                      P &= \sqrt{\frac{10^4}{5}}\\
                                      P &= 45
\end{align*}
$$
So, when P=45, adding new processors result in a slowdonw.


## Problem 3
+ Subsection a
> Because speed up $S_p(n)$：
$$S_p(n) = \frac{1}{f+\frac{1-f}{p}}$$
We can get $f$: 
$$f = \frac{1-950}{1000} = 0.05$$
So:
$$
\begin{align*}
  & S_p(10) = 6.897    \\
  & S_p(100) = 16.867  \\
  & S_p(1000) = 19.627 \\
\end{align*}
$$
And the Upper limit $S_{ul}$:
$$S_{ul} =\frac{1}{f} = 20$$

+ Subsection b
> Because：
$$S_p(n) = \frac{T^*(n)}{T_p(n)}$$
And:
$$
\begin{align*}
  T^*(n) &= 1000 \\
  T_p(n) &= T_{new}+ T_{fixed} + \frac{T_{computaion}}{P} \\
         &= 10 + 0.1 * P + 50 + \frac{950}{P}
\end{align*}
$$
So for P=10,100,100:
$$
\begin{align*}
  & S_p(10) = 6.41 \\
  & S_p(100) = 12.578 \\
  & S_p(1000) = 6.213
\end{align*}
$$

+ Subsection c
> $T=1000$, and assume the problem size is $N$. For $P$ processors：
$$T_p = 50 + \frac{N}{P}*10 = 1000$$
So:
$$
\begin{align*}
  & N_p(10)= 95       \\
  & N_p(100)= 9500    \\
  & N_p(1000)= 95000
\end{align*}
$$

When overheads of thread creation are taken into account:  
$$T_p = 50 +\frac{N}{P}*10 +  T_{ThreadCreate}= 1000$$
Here when $T_{ThreadCreate}= 10sec + 0.1sec + 0.1sec * P$:
$$T_p = 50 +\frac{N}{P}*10 + 10sec +  0.1sec * P = 1000$$
Solve this equation, we can get:
$$N=\frac{(940-0.1*P)*P}{10}$$
So:
$$
\begin{align*}
  & N_p(10)=  939  \\
  & N_p(100)= 9300 \\
  & N_p(1000)= 84000
\end{align*}
$$

## Problem 4
### a
P=8, B=16G/s  
$At_{max} = 0.5GFLOPS * P = 4GFLOPS$  
$AI_{max}= At_{max} / B = 4/16 = 0.25 FLOPS/Byte $  

Computer Float for phase b is: $16G/8s = 2GFlops < At_{max}$

<!-- $AI_{b} = 0.4 Flops/Byte$
$IO= 2GFlops/AI_b = 5$ -->
So they should further optimize the algortihm.
<!-- computer bound -->
### b
Computer Flop for phase b is: $16G/4s = 4GFlops = At_{max}$  
So Compute FLOPS is the nex the bottleneck.

<!-- ?? compute FLOPS-->
### c
$At_{max} = 0.5GFLOPS * P = 8GFLOPS$  
Memory is the bottlnneck.

### d
<!-- todo -->

## Problem 5

### a
$E_A = CPI_A * (10*n^2) + (10^3 *n) * CPI_{ci}$  
$E_B = CPI_B * (10^3 * n * log_2(n)) + (10^3 *n) * CPI_{ci}$  

When n = $10^3$  
$E_A= 11000000 $  
$E_B= 15948676$  

When n =$10^4$  
$E_A= 1010000000 $  
$E_B= 209315685 $

### b
$MFLOPS = (FloatOperations) / Time $  
$MFLOPS_A= (10*n^2) / E_A* t_{cycle} ) $ 

$MFLOPS_B= (10^3 * n * log_2(n)) / E_B * t_{cycle}$
<!-- markdown -->
### c
cache miss latency time for A:
$E_{Amiss}=3.5*10^4*n *0.01 * 100ns $  
for B:  
$E_{Bmiss} 5*10^4*n *0.02 *100ns$  
When n = $10^3$  
$E_{A*} = E_A + E_{Amiss}$  
$E_{B*} = E_B + E_{Bmiss}$

$C=E_{A*}*P$

$efficiency = \frac{T^{*}}{C{*}}$