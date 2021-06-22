# lecture11-NN
Similarities and differences between GMs and NNs
![](Pasted%20image%2020210601114432.png)
![](Pasted%20image%2020210601123757.png)




I: Restricted Boltzmann Machines
![](Pasted%20image%2020210601161434.png)
![](Pasted%20image%2020210601161450.png)
![](Pasted%20image%2020210601161458.png)


II: Sigmoid Belief Networks
![](Pasted%20image%2020210601163509.png)
回忆baysian network中的head-to-head，当下一层observed时，上一层会变得dependent。因此整个网络很复杂，因为每个layer之内都是coupled的
![](Pasted%20image%2020210601163609.png)


"RBMs are infinite belief networks"
下面把RBM和Sigmoid Belief Network联系起来：
当我们在RBM中计算梯度时，要计算对于joint的expectation，这就要用到sample来简化计算。
假设我们使用gibbs-sampling，我们是在交替condition on 两组变量，hidden和observed。每算一次joint，我们都可以看作要进行infinite次这样的交替计算。
![](Pasted%20image%2020210601164240.png)
因此我们可以把RBM中对于joint的sample，看作是一个infinite layer的Sigmoid Belief Network的forward propagation。
但是需要注意的上面的infinite次condition on，都发生在某一次计算joint的过程当中，没有更新w，因此w是不变的，因此
![](Pasted%20image%2020210601164412.png)
RBM对应的不是任意的Sigmoid Belief Network，在这个network中每一层的参数都一样。
![](Pasted%20image%2020210601164852.png)
当我们来了新的data，可以看作是在原本的RBM或者原本的network上直接新增一组layer



III: Deep Belief Nets
![](Pasted%20image%2020210601171513.png)
![](Pasted%20image%2020210601171538.png)
![](Pasted%20image%2020210601195405.png)
![](Pasted%20image%2020210601195514.png)
![](Pasted%20image%2020210601195521.png)



Deep Boltzmann Machines的思想也差不多，也是layer wise，这里就跳过了
![](Pasted%20image%2020210601195543.png)


***
![](Pasted%20image%2020210601195758.png)
可以看到back-propagation的error很大，并没有inference出hidden variable真正的分布。然而其实实践中并不怎么关注这个。


***
我们提到在RBM的MCMC中，在经过infinite次sample之后，我们能得到true distribution。
在优化算法中，我们对unroll RBM得到的DBN经过infinite更新梯度之后，就有希望逼近true distribution。站在这个过程中，有两点要注意：所有的w时固定的；每层只计算一次，然后进入下一层。


但其实**在实际中，我们对DBN每层进行多次计算，然后最终使用finite层**。
打个比方，我们不是直接修一条到Rome的路，而是先修通往A的路，把路修到perfect，再修到B,C,D... 最终甚至可能并不关注是否到了Rome。但这中间的任何一段都能作为一个单独的task，有一定用处。
![](Pasted%20image%2020210601202413.png)
换句话说，我们通过一个step之内的多次计算optimize了求梯度这一过程。再换句话说，**we optimized the optimization**.
我们可以从上面这个角度来justify 只有finite steps的deep learning。

![](Pasted%20image%2020210601202951.png)
从这个角度，我们其实可以把graphic model的inference过程展开，从backprop的角度理解整个问题