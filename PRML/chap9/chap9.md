# 9. Mixture Models and EM

* As well as providing a framework for building more complex probability distributions, mixture models can also be used to cluster data
* A general technique for finding maximum likelihood estimators in latent variable models is the expectation-maximization (EM) algorithm. 
* 在很多应用中，Gaussian mixture models的参数都是用EM得到的，然而MLE是有缺陷的。chapter10会介绍Variational inference. Variational inference的计算量不会比EM多很多，解决了MLE的问题，而且allowing the number of components in the mixture to be inferred automatically from the data

## 9.1. K-means Clustering
目标是把N个D维的observed data x分为K个cluster，这里假定K已知。

我们引入K个D维向量$\mu_k$用于描述每个cluster的中心
我们想要找到一种对data的分配，使得the sum of the squares of the distances of each data point to its closest vector $\mu_k$,is a minimum

对每一个x，都用一个K维onehot表示这个data所属的cluster
![](Pasted%20image%2020210428160037.png)

然后就可以定义一个objective function，或者叫distortion measure：
![](Pasted%20image%2020210428160158.png)
Our goal is to find values for the $\{r_{nk}\}$ and the $\{\mu_{k}\}$ so as to minimize J.

我们可以用一个iterative procedure，每个iteration分为两步，分别对$r_{nk}$ 和 $\mu_{k}$优化

First we choose some initial values for the $\mu_{k}$. 
Then in the first phase we minimize J with respect to the $r_{nk}$, keeping the $\mu_{k}$ fixed. 
In the second phase we minimize J with respect to the $\mu_{k}$, keeping $r_{nk}$ fixed. 
This two-stage optimization is then repeated until convergence. 

之后我们会看到对$r_{nk}$和$\mu_{k}$分别对应E step和M step

Consider first the determination of the $r_{nk}$
we simply assign the $n^{th}$ data point to the closest cluster centre. More formally, this can be expressed as
![](Pasted%20image%2020210428161251.png)

Now consider the optimization of the $\mu_{k}$ with the $r_{nk}$ held fixed.
直接对μ求导可得
![](Pasted%20image%2020210428161415.png)
![](Pasted%20image%2020210428161427.png)
$\mu_{k}$ equal to the mean of all of the data points xn assigned to cluster k

Because each phase reduces the value of the objective function J, convergence of the algorithm is assured. However, it may converge to a local rather than global minimum of J.


把squared Euclidean distance推广到general dissimilarity，就得到了K-medoids algorithm
![](Pasted%20image%2020210428162647.png)
为了简化模型，会把$\mu$设为每个cluster中的某个data

注意在K-Means中是硬分类，每笔data只能分配给唯一的一个cluster


## 9.2. Mixtures of Gaussians

We now turn to a formulation of Gaussian mixtures in terms of discrete **latent** variables.

Gaussian mixture distribution可以写成linear superposition of Gaussians
![](Pasted%20image%2020210428170711.png)

***

下面我们从latent variable的角度引出上面这个式子：
引入一个1-of-K的变量$\mathrm{z}$, in which a particular element zk is equal to 1 and all other elements are equal to 0

定义一个x与z的joint，in terms of a marginal distribution p(z) and a conditional distribution p(x|z)
这样就可以得到graph：
![](Pasted%20image%2020210428220840.png)

然后我们把z的marginal用一组系数$\pi_k$表示
![](Pasted%20image%2020210428221026.png)
![](Pasted%20image%2020210428221040.png)
注意这里$\pi_k$要满足概率的性质（非负，和为一）

同时，我们把x在z=1的conditional设为gaussian：
![](Pasted%20image%2020210428221635.png)
合起来可以写成
![](Pasted%20image%2020210428221711.png)

The joint distribution is given by p(z)p(x|z), 
在joint上对z进行summation可以得到marginal p(x):
![](Pasted%20image%2020210428221858.png)

乍一看引入z好像没什么意义，However, we are now able to work with the **joint distribution p(x, z) instead of the marginal distribution p(x)**, and this will lead to significant **simplifications**, most notably through the introduction of the expectation-maximization (EM) algorithm

***

现在再定义一个比较重要的量：
the conditional probability of z given x, $\gamma(z_k)$
$\gamma(z_k)$可以直接用bayes theorem求
![](Pasted%20image%2020210428222852.png)
We shall view πk as the prior probability of zk =1, and the quantity γ(zk) as the corresponding posterior probability once we have observed x
As we shall see later, γ(zk) can also be viewed as the **responsibility** that component k takes for ‘explaining’ the observation x.
***
在mixture gaussian上generate random samples的方法：
之前提到的ancestral sampling：
先在marginal p(z)上生成一个value of z，记作$\hat{z}$，
然后用这个$\hat{z}$，在conditional$p(x|\hat{z})$上generate一个x

![](Pasted%20image%2020210428225623.png)
根据prior $\pi_k$可以得到图a，根据posterior $\gamma(z_k)$可以得到图c

