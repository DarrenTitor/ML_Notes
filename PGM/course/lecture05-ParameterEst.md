![](Pasted%20image%2020210525174708.png)
Note:
natual parameter和sufficient statistic是最重要的部分，它们之间做的是inner product操作，是linear dependent的。
重要性：关于η的parameter estimation只与T(x)有关

![](Pasted%20image%2020210526131848.png)
![](Pasted%20image%2020210526131925.png)

我们可以通过对A求导得到T(x)的moment：
然后moment parameter也往往与moment有关，因此这个性质有助于我们进行parameter estimation
![](Pasted%20image%2020210526132014.png)
![](Pasted%20image%2020210526132028.png)

![](Pasted%20image%2020210526132342.png)
moment parameter μ可以由natural parameter A得到，
然后观察二阶导数，因为它对应方差(大于0)，因此A是一个凸函数(一阶导数单调递增)。因此一阶导数所对应的μ可以与η一一对应。
![](Pasted%20image%2020210526132555.png)
因此我们可以把η和μ之间建立一个1to1的函数

在对exponential family的MLE中，我们们可以直接运用上面$\frac{dA(\eta)}{d\eta}=\mu$ 这一性质，简便地得到$\mu_{MLE}$.
![](Pasted%20image%2020210526133205.png)

***
下面来讲解把exponential family统一到natural parameter的作用：

因为data和parameter是linear dependent的，这一变换可以explicitly展示出data要像表现出这个distribution要经过怎样的transformation。(比如是否需要保存所有的data)


![](Pasted%20image%2020210526135806.png)
如何描述Y的分布？
通常对于X施加一个response function f()，然后对于$\mu=f(X)$施加一个exponential family，来描述Y
对于linear regression，f为$\theta^TX+b$
对于logistic regression，f为$\frac{1}{1+e^{-\theta^TX}}$

我们把conditional mean μ设为response f(x)，$E_p(Y)=\mu$
![](Pasted%20image%2020210526141403.png)

如果我们把f和$\psi$设为逆函数，则此时$\theta^TX$就对应着$\eta$
![](Pasted%20image%2020210526141551.png)

注意下面这些model使用的都是canonical response function
![](Pasted%20image%2020210526141636.png)
也就是说，隐藏的f和ψ就是上图中的形式，只不过最后抵消了，然后就有$\eta(x)=\theta^Tx$.

![](Pasted%20image%2020210526143357.png)
至于这里虽然上面那些distribution都对应着$\eta(x)=\theta^Tx$，但是在上图中的likelihood中，h(y)和A(η)还是不同的，可以通过这些体现出dist的不同。

当使用牛顿法时，利用二阶导数，此时我们必须知道$\psi$，没法抵消掉了，好在可以利用之前的那张表找到常见的dist对应的$\psi$

![](Pasted%20image%2020210526144227.png)
![](Pasted%20image%2020210526144239.png)


![](Pasted%20image%2020210526144741.png)

![](Pasted%20image%2020210526144832.png)
注意上图中如果一个node有两个parent，那么这两个node的影响往往还是可以写成相乘的关系，那么在log likelihood中就成为了相加的关系。

## EM

用于partial observed model
**In the e-step, we replace the sufficient statistic of hidden variable with their expection**
In the m-step, we just treat everything as observed，然后进行MLE

