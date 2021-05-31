# lecture02-MRFrepresentation
![](Pasted%20image%2020210523130415.png)
joint可以表示为每个clique上的potential的乘积，再加以正则化
potential是pre-probabilistic的，也就是没有正则化


![](Pasted%20image%2020210523200631.png)
H为一个无向图。如果P是一个Gibbs-distribution over H，也就是说P的factor不是任选的，而是选自graph H中的clique，然后再加以normalization，那么根据soundness theorem，**H is an I-map of P**。换句话说，H所能表示的东西，是P所能表示的东西的子集。这也就方便我们从一个graph H中，得到它的distribution(P)，而且可以保证至少不会损失independences。

![](Pasted%20image%2020210524094328.png)

Independence properties:
![](Pasted%20image%2020210524094459.png)
![](Pasted%20image%2020210524094623.png)
![](Pasted%20image%2020210524094656.png)
![](Pasted%20image%2020210524094715.png)





总结：
independence set是graph中的first citizen，
然后我们利用gibbs distribution从independence set中提取出概率分布


反之：
![](Pasted%20image%2020210523200149.png)
如果有三个variable setX,Y,Z，并不满足global markov property，我们仍然可以建立**某些P**使得他们可以在与之对应的图H上表现得independent。(比如，在特定的取值下，有可能两个变量之间可以看作独立，但稍微改一下数字，可能就不满足独立的式子了)


最后，我们用下面这个定理来解释Gibbs distribution的必要性：
![](Pasted%20image%2020210523201129.png)

当H和P完全等价的时候，称H为perfect map(很少见)
![](Pasted%20image%2020210523201342.png)

因为potential要求positive，所以从实践的角度我们把哦i藤条了定义为指数形式。
![](Pasted%20image%2020210523201729.png)



![](Pasted%20image%2020210523202342.png)
这个模型的作用：在后面的章节中，我们可以通过这个模型，从data中learn出一个graph。


当data是sparse的，对于boltzman machine，就有很多参数为0，对应到graph中，就有很多edge是缺失的。这就得到了ising-model，它其实是boltzman machine的特殊情况。
![](Pasted%20image%2020210523202925.png)

![](Pasted%20image%2020210523203617.png)
可以看到RBM中用到了singleton energy和pairwise energy。
RBM中的weight θ是未知的，我们要从data中学习出这些weight。

性质：
1. visible unit没有observed时，hidden factor都是dependent的
2. 当所有的visible unit都observed时，hidden factor之间独立。反之也成立。

(可以用head to head来解释)

![](Pasted%20image%2020210523204731.png)
可以把pairwise-energy定义为singleton energy的乘积再乘上权重，来进一步简化模型。

