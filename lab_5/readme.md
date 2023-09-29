### net structure

Lay 1:  b_x * W1 (784) -> a1 (128)
Lay 2:  a1 * W2 -> a2
Lay 3:  a2 * W3 -> softmax -> yhat

```
# 1 core, task 1.1
Iteration #: 999
Iteration Time: 0.0285755s
Forward and Backward Time(s) per epoch:0.028241 98.8294% time spend at dot in an epoch
Loss: 0.886171

# 2 core, task 1.1
Iteration #: 999
Iteration Time: 0.0140945s
Forward and Backward Time(s) per epoch:0.013715 97.3071% time spend at dot in an epoch
Loss: 1.12925


Iteration #: 999
Iteration Time: 0.0062754s
Forward and Backward Time(s) per epoch:0.005907 94.1295% time spend at dot in an epoch
Loss: 3.59998
```