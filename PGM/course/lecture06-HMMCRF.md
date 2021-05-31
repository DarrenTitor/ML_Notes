## HMM

之前讲的mixture model中，我们一次拥有全部的data
现在假设我们逐个得到数据：
![](Pasted%20image%2020210526202401.png)


![](Pasted%20image%2020210526233524.png)
注意下图中的推导，由于markov property，观测到$y_t$之后，t之前的x和t之后的x就是条件独立的
![](Pasted%20image%2020210526221809.png)







如果我们想inference某一个时刻的概率given一整个sequence，那么可以直接把前向和后向的message α和β直接相乘
![](Pasted%20image%2020210526223332.png)
如果我们想得到某一个pair的joint，given整个sequence：$p(y_t^i,y_{t+1}^i|x_1,...,x_T)$
![](Pasted%20image%2020210526223700.png)


我们现在可以算单一latent variable的MPA
![](Pasted%20image%2020210526224613.png)

如果我们想求一个pair的joint的MPA，那么不能分别求t和t+1的MPA，因为这joint的configuration和单独的configuration是不一样的，比如下面这个例子
![](Pasted%20image%2020210526224819.png)

如果我们想求所有hidden state的posterior，理论上我们就是想求$p(y_1,...,y_T|x_1,...,x_T)$，但是这个东西没法存。
![](Pasted%20image%2020210526233433.png)

![](Pasted%20image%2020210526233235.png)

![](Pasted%20image%2020210526233645.png)
![](Pasted%20image%2020210526233730.png)
![](Pasted%20image%2020210526233203.png)
![](Pasted%20image%2020210526233119.png)
MLE的缺点是over fitting，当某一个组合在data中没有出现，就会导致性能很差。因此有pseudocounting，也就是增加一定的数目，使得count不会为0.
而这相当于施加conjugate prior
![](Pasted%20image%2020210526232900.png)
![](Pasted%20image%2020210526232812.png)







## CRF

![](Pasted%20image%2020210527141206.png)
![](Pasted%20image%2020210527141153.png)
![](Pasted%20image%2020210527141226.png)

![](Pasted%20image%2020210527141135.png)




我们发现在learning中，把likelihood对于参数求导之后会发现一个奇怪的结果。原本我们是fully observed的，x和y都知道，但在梯度中，第一项是counting的部分，第二项是counting的期望，而这里我们要假装我们不知道y，然后去求这个期望。也就是说我们其实要做一个inference，为不光是简单的counting。
![](Pasted%20image%2020210527141116.png)

![](Pasted%20image%2020210527141438.png)
最终的求解可以用gradient ascent，其中期望可以用sum-product。
而整体上看，其实相当于使用了一个EM，尽管y其实是observed的



