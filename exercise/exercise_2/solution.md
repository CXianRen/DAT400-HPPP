# Session 2 Performance Analysis Problems
+ Name: Hongguang Chen
+ CID: chenhon
## Problem 1
### a
$E_t = T_p = 1000/P(compuations) + 2 (synchro) + 10 (data exchange) = 125 + 2 + 10 = 137 s$
$C_p = P*T_p = 137*8= 1096 s$
$efficiency = T^* / C_p = T* / (P * T_p ) = 1000/1096 = 0.91$

### b
$E_t = T_p = max(180, (1000-180)/7) + 2 + 10  =  180 + 2 + 10 = 192$
$C_p = p* T_p = 192 * 8 = 1536$
$efficiency = T^* / C_p = 1000/ 1536 = 0.65$


## Problem 2
N=1,000,000 and P=8
### a
$E_{static} = N/P * 10ns + T_{create} = 10^6/8 * 10* 10^{-3}us + 50us + 8*5us =  1250+ 90 = 1340 us$ 

$E_{self-scheduling} = N/P*(B+\sigma) + T_{create} = 10^6/8 * (10ns + 200ns) + 90us = 26250 + 90 = 26340 us $

chunk size of k =100,
$E_{chunk} = \frac{N}{k*P}*(k*B + \sigma) + T_{create} = 10^6/(100*8)*(100*10ns + 200ns) + 90us = 15090 us $

### b
P=16
$E_{static} = N/P * 10ns + T_{create} = 10^6/16 * 10* 10^{-3}us + 50us + 16*5us  = 755 us$ 

$E_{self-scheduling} = N/P*(B+\sigma) + T_{create} = 10^6/16 * (10ns + 200ns) + 130us  = 13255 us $

$E_{chunk} = \frac{N}{k*P}*(k*B + \sigma) + T_{create} = 10^6/(100*16)*(100*10ns + 200ns) + 130us = 7630 us $

P=32
$E_{static} = 482.5 us$
$E_{self-scheduling} = 6732.5 us $
$E_{chunk} = 3920 us$

### c
Becasue:$S=T^*/T_p$
$T_p = N/P * 10ns + T_{create}$
$T^* = N* 10ns $
When:
$$S<=1$$
Then:
$$T^*/T_p = 1$$
Then:
$$N*10ns = N*10ns/P+ 50us + P* 5us=$$
Then:
$$ 2*10^3 = (2*10^3)/P + P$$
let $k =2*10^3$:
$$ k = k/P +P  → kP-P^2$$
differentiate with respect to P:
$$ K=2P$$
so P=1000.

## Problem 3
### a 
$S_p(n) = \frac{1}{f+\frac{1-f}{p}}$
$f = \frac{1-950}{1000} = 0.05$
So:
$S_p(10) = 6.897$
$S_p(100) = 16.867$
$S_p(1000) = 19.627$

$UpperLimit =\frac{1}{f} = 20$

### b
Because：
$$S_p(n) = \frac{T^*(n)}{T_p(n)}$$
And:
$$T^*(n) = 1000$$
$$T_p(n) = T_{new}+ T_{fixed} + \frac{T_{computaion}}{P} * = 10 + 0.1 * P + 50 + \frac{950}{P}$$
So for P=10,100,100:
$S_p(10) = 6.41$
$S_p(100) = 12.578$
$S_p(1000) = 6.213$

### c
T=1000, and assume size is N, 
For P processors：
$$T_p = 50 + \frac{N}{P}*10 = 1000$$
So:
$N_p(10)= 950$
$N_p(100)= 9500$
$N_p(1000)= 95000$

When overheads of thread creation are taken into account:
$$T_p = 50 + T_{ThreadCreate}+\frac{N}{P}*10 = 1000$$
the result will reply on how to calculate the overhead of thread creation.

## Problem 4


## Problem 5