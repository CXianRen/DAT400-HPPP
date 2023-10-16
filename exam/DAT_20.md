# Problem 1
(a) If there is a task required so much computation over a single machine can provided, for example, the sum of compuation resource it required over a thounsand single PCs. Thus if we want to run it in parallel effectivly, we have to run it in a super computer set (many PC connected by network, which is been see as non-shared memeory system), using MPI or similar library to dispatch subset of task into each single PC, to sycn the state, to collect the result.  

(b) MPI provides both blocking and unblocking APIs to send and to receive msg among processes, eassily casuing waiting for a message which will never be sent. While for openMP, it divides a task just with the compiler directives, which had hided the details for developer, especially for the default access right of the local variables.

(c) 
```cpp
// OpenMP 


// MPI

```
