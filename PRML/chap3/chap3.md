## 3.1. Linear Basis Function Models
线性回归就是普通的线性组合，有很大的限制，因此这里拓展为linear combinations of fixed nonlinear functions of the input variables
![](Pasted%20image%2020210419191436.png)
![](Pasted%20image%2020210419191624.png)
where $φ_j(x)$ are known as basis functions


几个basis：
Gaussian basis：
![](Pasted%20image%2020210419211558.png)
(although it should be noted that they are not required to have a probabilistic interpretation)

sigmoidal basis:
![](Pasted%20image%2020210419211731.png)
Equivalently, we can use the ‘tanh’ function because this is related to the logistic sigmoid by tanh(a)=2σ(a) − 1

(在这章，并不具体特指某种basis，因此简记φ(x)= x)

### 3.1.1 Maximum likelihood and least squares

Note that the Gaussian noise assumption implies that the conditional distribution of t given x is **unimodal**, which may be inappropriate for some applications. An extension to mixtures of conditional Gaussian distributions, which permit multimodal conditional distributions, will be discussed in Section 14.5.1.

把上边的式子改写成向量形式：Likelihood function:
![](Pasted%20image%2020210419220550.png)
Note that in supervised learning problems such as regression (and classification), we are not seeking to model the distribution of the input variables.
因此X永远在condition这边，因此省略
![](Pasted%20image%2020210420090558.png)
![](Pasted%20image%2020210420090646.png)

这时梯度为
![](Pasted%20image%2020210420090833.png)
解上面这个式子，可以得到正规方程
![](Pasted%20image%2020210420091112.png)

pseudo inverse：
当矩阵is square and invertible，pseudo inverse转化为inverse
![](Pasted%20image%2020210420091243.png)

### 3.1.3 Sequential learning
if the data set is sufficiently large, it may be worthwhile to use **sequential algorithms, also known as on-line algorithms**,
in which the data points are considered one at a time, and the model parameters updated after each such presentation.
-> stochastic gradient descent, also known as sequential gradient descent

### 3.1.4 Regularized least squares
L2对应的solution：
![](Pasted%20image%2020210420103453.png)

正则化是为了约束w，因此可以写成
![](Pasted%20image%2020210420104458.png)
这样就可以用Lagrange：
![](Pasted%20image%2020210420104541.png)

quadratic，约束为仿射，满足KKT，因此：
![](Pasted%20image%2020210420104721.png)
可以把上面这个式子画到w各维组成的空间中，蓝色的为error的等高线，就能得到下面这张熟悉的图：
![](Pasted%20image%2020210420104911.png)

### 3.1.5 Multiple outputs

## 3.2. The Bias-Variance Decomposition

h(x)是t的条件期望，这个在1.5.5重已经提到过
Loss的期望可以分解为以下两项，第一项y(x)与model有关，而第二项只取决于数据自身的noise
![](Pasted%20image%2020210420192405.png)


下面来探讨model自身的uncertainty：
* 如果是bayesian方法，model的不确定性由**posterior** distribution over w决定
* 如果是frequentist方法：
	A frequentist treatment, however, involves making a point estimate of w based on the data set D, and tries instead to interpret the uncertainty of this estimate through the following thought experiment. Suppose we had a large number of data sets each of size N and each drawn independently from the distribution p(t, x). 
	For any given data set D, we can run our learning algorithm and obtain a prediction function y(x; D). Different data sets from the ensemble will give different functions and consequently different values of the squared loss. **The performance of a particular learning algorithm is then assessed by taking the average over this ensemble of data sets.**
	
对于上面式子的第一项进行配凑，再对于D取期望，可以得到
![](Pasted%20image%2020210420194707.png)

![](Pasted%20image%2020210420195208.png)

![](Pasted%20image%2020210420195312.png)

## 3.3. Bayesian Linear Regression

### 3.3.1 Parameter distribution

For the moment, we shall treat the noise precision parameter β as a known constant. 
这里选择gausiian是因为，观察
![](Pasted%20image%2020210420212409.png)
可以发现p(t|w) defined by (3.10) is the exponential of a quadratic function of w
因此共轭的先验就选择Gaussian
![](Pasted%20image%2020210420212340.png)

Due to the choice of a conjugate Gaussian prior distribution, the posterior will also be Gaussian

这里使用第二章的conditional of gaussian的结论可以直接写出prior与likelihood相乘之后归一化得到的posterior的表达式
![](Pasted%20image%2020210420213133.png)
![](Pasted%20image%2020210420213206.png)

因为gaussian的最值就等于均值，因此 $w_{MAP} = w_N$
同时，可以观察到，当prior的$variance\to\infty$，后验的$m_N$会转化为MLE的结果，也就是之前提到的正规方程
![](Pasted%20image%2020210420214524.png)


本章下文中，讨论的是zero-mean isotropic(各向同性) Gaussian governed by a single precision parameter α
![](Pasted%20image%2020210420214737.png)

log of posterior:
![](Pasted%20image%2020210420214840.png)
最大化log posterior等价于最小化sum-of-squares with regularization term λ = α/β


### 3.3.2 Predictive distribution
![](Pasted%20image%2020210420222821.png)

利用上边的2.115的margin结论，可以直接得出output的分布：
![](Pasted%20image%2020210420223100.png)

在variance的表达式中，第一项是因为数据的noise，第二项是因为w的uncertainty。而noise和w是独立的，因此是相加的关系


缺陷：
If we used localized basis functions such as Gaussians, then in regions away from the basis function centres, the contribution from the second term in the predictive variance (3.59) will go to zero, leaving only the noise contribution β−1. 

Thus, the model becomes very confident in its predictions when extrapolating outside the region occupied by the basis functions, which is generally an undesirable behaviour. This problem can be avoided by adopting an alternative Bayesian approach to regression known as a **Gaussian process**.

### 3.3.3 Equivalent kernel

the predictive mean can be written in the form
![](Pasted%20image%2020210422092728.png)

这个式子可以写成kernel的形式，解释为所有数据的线性组合：
![](Pasted%20image%2020210422092938.png)
这个的kernel叫做 smoother matrix or the equivalent kernel
Regression functions, such as this, which make predictions by taking linear combinations of the training set target values are known as **linear smoothers**.

以上我们先找了basis function，然后找到等价的kernel。其实我们还可以直接定义一个local kernel，直接用已知的所有数据做预测。这就是Gaussian processes

![](Pasted%20image%2020210422094134.png)

equivalent kernel和其他所有kernel都有一个重要的性质，it can be expressed in the form an inner product with respect to a vector ψ(x) of nonlinear functions
![](Pasted%20image%2020210422094657.png)

## 3.5. The Evidence Approximation

在fully Bayesian treatment中，我们要为超参数α和β设先验，and make predictions by marginalizing with respect to these hyperparameters as well as with respect to the parameters w. 但是实际上这样的积分是intractable的。

Here we discuss an approximation in which we set the hyperparameters to specific values determined by maximizing the marginal likelihood function obtained by first integrating over the parameters w
（我们先对w积分，然后用MLE确定hyperparameters）

这种方法有好几个名字：empirical Bayes, type 2 maximum likelihood, generalized maximum likelihood, evidence approximation


精确的写法应该是这样
![](Pasted%20image%2020210422100421.png)
但如果p(α, β|t)中，α, β都sharply peaked，那么就可以把这两个参数固定，近似为下面的式子
![](Pasted%20image%2020210422100529.png)

α, β的后验由下面这个式子决定，
![](Pasted%20image%2020210422100712.png)
If the prior is relatively flat, then in the evidence framework the values of α and β 就可以通过maxmize likelihood 得到，这里的MLE是通过所有的training data直接得到α, β，而不用cross-validation
Recall that the ratio α/β is analogous to a regularization parameter.

***

Returning to the evidence framework, we note that there are two approaches that we can take to the maximization of the log evidence

* evaluate the evidence function analytically and then set its derivative equal to zero to obtain re-estimation equations for α and β, which we shall do in Section 3.5.2
* use a technique called the expectation maximization (EM) algorithm




The marginal likelihood function p(t|α, β) is obtained by integrating over the weight parameters w, so that
![](Pasted%20image%2020210422104412.png)
其中，代入
![](Pasted%20image%2020210422104248.png)
![](Pasted%20image%2020210422104308.png)
可以得到
![](Pasted%20image%2020210422104456.png)
![](Pasted%20image%2020210422104504.png)

