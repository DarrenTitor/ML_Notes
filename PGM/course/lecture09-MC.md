# lecture09-MC


![](Pasted%20image%2020210530125650.png)
用sample的方法表达一个distribution，就是保存一堆点以及其对应的函数值。求期望的话就对应这些点的期望。

![](Pasted%20image%2020210530162045.png)
***
![](Pasted%20image%2020210530162819.png)
我们想近似一个分布Π，但是我们只用它的unnormalized部分Π'(X)。
然后我们从一个更简单的分布Q(X)中采样，计算 $\Pi(x^*)/kQ(x^*)$，k是一个定值使上面这个值处在0,1之间。

然后通过上图中的证明，可以看到：我们最初采样的x其实只有一部分能保留下来，而保留下来的这些采样点所对应的概率正好等于Π

缺点：
![](Pasted%20image%2020210530164220.png)
在高维情况下，即使我们的Q与真正的P很接近，也会使得k非常大


***

![](Pasted%20image%2020210530170809.png)
把 $\frac{P(x^*)}{Q(x^*)}$ 当作weight，为每一个采样点都保存这样一个weight。
缺点：需要使用normalized P，对于markov random field，用不了。

![](Pasted%20image%2020210530171633.png)
我们可以用unnormalized P'代替P，α可以用上面的公式求出，此时我们经过推导可以得到一个normalized的weight

缺点：之前的版本可以用作online algorithm，每一个点的weight都可以立即得到。但是在这个版本，我们需要sample苏够多的点之后才能得到weight


***
importance sampling可能停在一个很坏的结果上
![](Pasted%20image%2020210530174319.png)
如果P和Q差别很大，大部分时候我们都在Q大的地方sample，但是这时P/Q很小，使得新的点的weight很小，对于整个结果没有什么影响，最后结果就很差。

![](Pasted%20image%2020210530184739.png)
通过在sample上再sample，P/Q小的部分就会有很小的几率再被sample了
![](Pasted%20image%2020210530184801.png)




![](Pasted%20image%2020210530192441.png)
![](Pasted%20image%2020210530192455.png)

***
![](Pasted%20image%2020210530192952.png)
![](Pasted%20image%2020210530193240.png)

Note：
**A usual choice is to let Q(x∣y) be a Gaussian distribution centered at y, so that points closer to  y are more likely to be visited next**


![](Pasted%20image%2020210530193306.png)


![](Pasted%20image%2020210530194108.png)


Notes:
人们经常丢掉前几千个sample
![](Pasted%20image%2020210530194224.png)
因为当t过小的时候，整个Q(x'|x)可能卡在一个bad区域，没有开始在整个空间运动

***
为什么会收敛？
![](Pasted%20image%2020210530194438.png)

关于收敛的证明需要用到markov chain：
![](Pasted%20image%2020210530195434.png)

在之前的MC中，每一个sample抽取自Q，但是它们最终被抽取的概率也等于P。因为Q可以被约掉，所以其实Q长什么样并没有很重要。
而在MCMC中，注意到我们的Q(x'|x)是一直在变化的，因此我们需要关注Q的状态。
![](Pasted%20image%2020210530200639.png)
![](Pasted%20image%2020210530200731.png)
![](Pasted%20image%2020210530200844.png)
关于stationary distributions，我们可以把它类比为矩阵乘法。
当系统不再变化，代表着 $\pi=T\cdot \pi$，可以看到$\pi$可以看作是eigenvector。而不是所有的matrix都有eigenvector。因此不是所有的transition都能lead to stationary，但是某些好的transition可能会导致stationary。


下面要提到几个MC的概念：
![](Pasted%20image%2020210530220037.png)
也就是说，可以保证从一个state可以在有限的step之后移动到另一个state。保证了我们是有可能达到stationary的。

![](Pasted%20image%2020210530220356.png)
MC可能从state i到state j，再从state j直接回到state i，而不是经过一个很长的cycle再回到state i。

![](Pasted%20image%2020210530220531.png)
然后定义一个概念，称满足上面两个性质的MC为ergodic的。

![](Pasted%20image%2020210530220708.png)
如果一个MC满足ergodic，那么它就可以达到stationary(如果有的话)

![](Pasted%20image%2020210530221441.png)
如果一个MC满足detailed balance，那么它就有stationary
***
所以，一个MC要想work，我们就要构造合适的T，使得MC有detailed property，然后这个MC就可以converge somewhere。最好是我们可以设计合适的T使得最终收敛到P。

Metropolis-Hastings收敛的证明：

![](Pasted%20image%2020210530222800.png)
![](Pasted%20image%2020210530224326.png)

![](Pasted%20image%2020210530224859.png)

***
Gibbs sampling:

idea:
My new example is all the same as the previous one except one dimention. And that dimention is draw from a conditional from the remaining dimentions. 
$x_i^{t}\sim P(x_i|x^{t-1}_{\{-i\}})$ 
![](Pasted%20image%2020210530230106.png)
然后，因为有markov blanket，draw next sample的计算会很简单
![](Pasted%20image%2020210530230234.png)

![](Pasted%20image%2020210530233407.png)


***
![](Pasted%20image%2020210531100207.png)
![](Pasted%20image%2020210531100304.png)
![](Pasted%20image%2020210531100322.png)

![](Pasted%20image%2020210531100930.png)
可以由此算出一个系数，由此我们可以知道，当我们sample了100个点时，其实相当于sample了多少个independent的点
![](Pasted%20image%2020210531101019.png)


![](Pasted%20image%2020210531101417.png)
其中一种方法是，比较多次MC的结果，如果结果差不多说明收敛了
![](Pasted%20image%2020210531101538.png)
或者画出log likelihood
![](Pasted%20image%2020210531101746.png)






![](Pasted%20image%2020210531101844.png)
![](Pasted%20image%2020210531103947.png)












