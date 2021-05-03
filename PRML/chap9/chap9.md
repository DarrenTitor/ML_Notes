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

### 9.2.1 Maximum likelihood
Suppose we have a data set of observations {x1,..., xN}, and we wish to model this data using a mixture of Gaussians
对应的latent variable可以写成一个NxK的矩阵Z
![](Pasted%20image%2020210429103842.png)
假设i.i.d，则可以写出log likelihood：
![](Pasted%20image%2020210429103717.png)
***
下面来探讨对mixture gaussian进行MLE所存在的问题：
For simplicity, consider a Gaussian mixture whose components have covariance matrices given by $\Sigma_k = \sigma^2_k\textbf{I}$

Suppose that one of the components of the mixture model, let us say the jth component, has its mean µj exactly equal to one of the data points so that µj = xn for some value of n. This data point will then contribute a term in the likelihood function of the form

![](Pasted%20image%2020210429104756.png)
$\sigma_j\to0$时，这一点对likelihood的贡献趋于无穷，likelihood趋于无穷

因此MLE不适用于Mixture Gaussian，因为such singularities will always be present and will occur whenever one of the Gaussian components ‘collapses’ onto a specific data point.

Recall that this problem did not arise in the case of a single Gaussian distribution. To understand the difference, note that if a single Gaussian collapses onto a data point it will contribute multiplicative factors to the likelihood function arising from the other data points and these factors will go to zero exponentially fast, giving an overall likelihood that goes to zero rather than infinity. 
However, once we have (at least) two components in the mixture, one of the components can have a finite variance and therefore assign finite probability to all of the data points while the other component can shrink onto one specific data point and thereby contribute an ever increasing additive value to the log likelihood


还有一个问题就是**identifiability**。for any given (nondegenerate) point in the space of parameter values there will be a further K!−1 additional points all of which give rise to exactly the same distribution.
简单来说，就是K组参数和K个component，K个萝卜K个坑，顺序无所谓结果，但是却对应着K！个solution

Mixture gaussian比单个gaussian更难计算的原因，其实就是(9.14)的log likelihood中，ln内有求和，这个是不能做简化的。因此求导之后不能直接得到解

### 9.2.2 EM for Gaussian mixtures

An elegant and powerful method for finding maximum likelihood solutions for models with latent variables is called the **expectation-maximization algorithm**, or EM algorithm 

将log likelihood对$\mu_k$求导
![](Pasted%20image%2020210430111653.png)
![](Pasted%20image%2020210430111833.png)
![](Pasted%20image%2020210430114303.png)
可以看到右边的其中一项就是$\gamma(z_{nk})$，也就是$p(z_k=1|x)$这个后验

multiplying by $\Sigma_k$(which we assume to be nonsingular)可以得到
![](Pasted%20image%2020210430112329.png)
We can interpret $N_k$ as the effective number of points assigned to cluster k.

可以看到，$\mu_k$是第k个gaussian的data的weighted sum，weight则是posterior probability γ(znk) that component k was responsible for generating xn

接下来用相同的思路对$Sigma_k$求导，making use of the result for the maximum likelihood solution for the covariance matrix of a single Gaussian, we obtain
![](Pasted%20image%2020210430115412.png)
![](Pasted%20image%2020210430115112.png)
again with each data point weighted by the corresponding posterior probability and with the denominator given by the effective number of points associated with the corresponding component.

最后我们要对$\pi_k$做MLE，要用到Lagrange，因为对于$\pi_k$是有约束的：
![](Pasted%20image%2020210430115707.png)

![](Pasted%20image%2020210430115757.png)

因此第k个gaussian的系数就是the average responsibility which that component takes for explaining the data points

需要强调的是，上面的三个式子并不是model的一个closed-form solution，
但是我们可以用iterative scheme来求解，
1. 首先选定means, covariances, and mixing coefficients的初始值
2. alternate between the following two updates that we shall call the E step and the M step
	1. In the expectation step, or E step, we use the current values for the parameters to evaluate the **posterior probabilities, or responsibilities**, given by (9.13)
	![](Pasted%20image%2020210501095057.png)
	2. We then use these probabilities in the maximization step, or M step, to re-estimate the **means, covariances, and mixing coefficients** using the results (9.17), (9.19), and (9.22)

注意在M step中我们先求$\mu$，再用这个$\mu$求$\Sigma$

EM的iteration数与每个循环中的计算量都远大于KMeans，因此常常先用Kmeans找初始值，再用gaussian mixture进一步求：The covariance matrices can conveniently be initialized to the sample covariances of the clusters found by the K-means algorithm, and the mixing coefficients can be set to the fractions of data points assigned to the respective clusters. 
强调：log likelihood function有多个local maxima, EM并不保证找到global maxima

## 9.3. An Alternative View of EM
**The goal of the EM algorithm is to find maximum likelihood solutions for models having latent variables**

We denote the set of all observed data by $\textbf{X}$, in which the $n^{th}$ row represents $\textbf{x}^T_n$, and similarly we denote the set of all latent variables by $\textbf{Z}$, with a corresponding row $\textbf{z}^T_n$. The set of all model parameters is denoted by $\theta$, and
so the log likelihood function is given by
![](Pasted%20image%2020210501103630.png)
对于continuous latent variables，只需要把求和换成积分

注意到summation出现在ln内，Even if the joint distribution p(X, Z|θ) belongs to the exponential family, the marginal distribution p(X|θ) typically does not as a result of this summation. 求和使得ln不能直接作用于joint，导致MLE的解很复杂

假设对于每个observation in X，我们都知道对应的Z，We shall call {X, Z} the **complete** data set, and we shall refer to the actual observed data X as **incomplete**。
![](Pasted%20image%2020210501104541.png)
（a是complete的，b是incomplete的）
我们可知，对于complete data set的log likelihood就是简单的ln p(X,Z|θ), **我们假设这个log likelihood的maximization是好求的**。

然而在实践中，我们没有complete data set，只有incomplete data set X
Our state of knowledge of the values of the latent variables in Z is given only by the posterior distribution p(Z|X, θ)

**Because we cannot use the complete-data log likelihood, we consider instead its expected value under the posterior distribution of the latent variable, which corresponds (as we shall see) to the E step of the EM algorithm.** 
上面这句很关键，我们是在表示complete-data log likelihood的期望，其中概率是posterior p(Z|X, θ)，value是 ln p(X, Z|θ)。也就是用incomplete的X的log likelihood在p(Z|X, θ)上的概率求期望。这个期望不是X的期望，而是人为定义的一个值，是对complete-data log likelihood的近似。而这计算这个期望的步骤对应E-step，而求出使这个期望最大的参数的过程对应M-step。
如果当前参数对应$\theta^{old}$，那么先后经历了一个Estep和Mstep之后，我们就得到了一个新的$\theta^{new}$

The use of the expectation may seem somewhat arbitrary. However, we shall see the motivation for this choice when we give a deeper treatment of EM in Section 9.4.

总结：
* In the E step, we use the current parameter values $\theta^{old}$ to find the posterior distribution of the latent variables given by $p(Z|X, \theta^{old})$.
	We then use this posterior distribution to find the expectation of the complete-data log likelihood evaluated for some general parameter value $\theta$. This expectation, denoted $\mathcal{Q} (\theta,\theta^{old})$, isgiven by
	![](Pasted%20image%2020210501122529.png)
* In the M step, we determine the revised parameter estimate $\theta^{new}$ by maximizing this function
	![](Pasted%20image%2020210501122813.png)
	
Note:
注意现在在$\mathcal{Q} (\theta,\theta^{old})$ 中，ln直接作用在joint上，ln中没有summation了，因此可以计算了

![](Pasted%20image%2020210501123432.png)
![](Pasted%20image%2020210501123447.png)

The EM algorithm can also be used to find MAP (maximum posterior) solutions for models in which a prior p(θ) is defined over the parameters. 
In this case the E step remains the same as in the maximum likelihood case, whereas in the M step the quantity to be maximized is given by $\mathcal{Q} (\theta,\theta^{old}) + ln p(\theta)$.


### 9.3.1 Gaussian mixtures revisited
EM的motivation：(incomplete不知道Z导致了ln(summation), 没法算. 而下面来展示如果是complete的话，就很好算)
现在假设我们不光知道X，还知道Z，也就是知道是哪个component产生了X。（注意这个Z是hard的onehot，与posterior的$\gamma(z_{nk})$区分开）
![](Pasted%20image%2020210501222803.png)
此时likelihood为：
![](Pasted%20image%2020210501222912.png)
此时可以看到，ln内部没有summation，而ln里的gaussian本身就是exponential family中的，因此很好算

（下面从隐变量的角度把GMM再推一下）
然而实际中并没有隐变量的值，因此我们把数据X在Z的后验上的期望看作是complete-data log likelihood

![](Pasted%20image%2020210501224610.png)
![](Pasted%20image%2020210501224619.png)
再加上bayes定理可以得到
![](Pasted%20image%2020210501224659.png)
![](Pasted%20image%2020210501224726.png)

与之前的硬算p(X)
![](Pasted%20image%2020210501224830.png)
相比，可以看到ln与求和交换了位置，因此好算

* the maximization with respect to the means and covariances:
	因为$z_{nk}$是onehot，因此(9.36)只是K个independent的gaussian相加，因此结果就是K个gaussian各自算各自的参数
* the maximization with respect to the mixing coefficients：
	因为有sum为1的约束，因此还是要用lagrange，得到
	![](Pasted%20image%2020210501231105.png)
	因此mixing coefficients are equal to the fractions of data points assigned to the corresponding components
	
那么问题来了，$z_{nk}$我们不知道，因此和前面一样，我们用bayes：（其实式子也和前面(9.35)是一个道理，因为posterior正比于joint）
![](Pasted%20image%2020210501232121.png)
观察这个式子我们可以发现。N个$z_n$是independent的，因此我们可以
![](Pasted%20image%2020210501233712.png)
![](Pasted%20image%2020210501233725.png)
(Mark，上面这个求期望没咋看懂)
The expected value of the complete-data log likelihood function is therefore given by
![](Pasted%20image%2020210501234157.png)


### 9.3.2 Relation to K-means
Kmeans是hard的，GMM是soft的，我们可以把kmeans看作是GMM的一个limit

假设GMM的每个component的gaussian，covariance  matrix都是$\epsilon I$, where $\epsilon$ is a variance parameter that is shared by all of the components
![](Pasted%20image%2020210501235619.png)
我们假设，$\epsilon$不需要在EM中求解，然后我们代入EM-GMM的结论：
可以得到当$\epsilon \to 0$时，em的解就是kmeans的解


## 9.4. The EM Algorithm in General
(proof that the EM algorithm indeed maximize the likelihood function)

我们的目标是maximize
![](Pasted%20image%2020210502102258.png)

假设p(X|θ)难算，但是complete-data likelihood function p(X, Z|θ)很简单
我们引入q(Z)作为Z的分布，然后可以看到对于任意的q(Z)，都有
![](Pasted%20image%2020210502102808.png)
因为有：
![](Pasted%20image%2020210502103517.png)

这里的KL散度可以理解为：用p(Z|X,θ)近似q(Z)带来的熵增
L(q,θ)则是一个q的functional，也是一个θ的function

因为KL(q||p)>=0,所以L(q, θ)<=ln p(X|θ)
也就是说**L(q, θ)是ln p(X|θ)的lower bound**
![](Pasted%20image%2020210502105420.png)
在E-step中，我们固定住$\theta^{old}$，关于q(Z)maxmize L(q, θ).因为p(X|θ)和q(Z)没关系，所以L(q, θ)能一直取到上限，也就是KL散度为0，q(Z)=p(Z|X,θ)的情况。**In this case, the lower bound will equal the log likelihood**
![](Pasted%20image%2020210502105611.png)

在M-step中，我们固定住q(Z)，关于θ对L(q, θ)最大化，从而得到$\theta^{new}$
此时lower bound L会变大，也就导致整个log likelihood变大
注意此时q(Z)不再等于p(Z|X,θ)，因为θ已经更新了，所以KL大于0(这里讨论的都是还没有收敛的情况)。**The increase in the log likelihood function is therefore greater than the increase in the lower bound**
![](Pasted%20image%2020210502110251.png)
我们可以把$q(Z)=p(Z|X,\theta^{old})$代入L的定义式，得到
![](Pasted%20image%2020210502110435.png)
(因为之前在GMM中我们已经定义过近似的complete log likelihood为)
![](Pasted%20image%2020210502110718.png)
再观察，在上面M-step中，θ只出现在ln内部，因此在joint p(X, Z|θ)为exponential family的情况下，计算就会比直接算p(X|θ)简单

***
用EM做MAP：p(θ|X)，其中设定prior p(θ)
和之前一样，我们先把我们的目标转换成joint
![](Pasted%20image%2020210502112053.png)

然后和之前一样，我们同样把KL散度拆出来，因为我们已知它是非负的：
![](Pasted%20image%2020210502112221.png)
where ln p(X) is a constant.
此时对这个式子进行EM，
在E-step中，对q优化的式子与之前一样
在M-step中，对θ优化的过程不变，只是在式子中多了一项lnp(θ)