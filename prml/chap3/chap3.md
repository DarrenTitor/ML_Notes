# chap3

## 3.1. Linear Basis Function Models

线性回归就是普通的线性组合，有很大的限制，因此这里拓展为linear combinations of fixed nonlinear functions of the input variables ![](../../.gitbook/assets/Pasted%20image%2020210419191436.png) ![](../../.gitbook/assets/Pasted%20image%2020210419191624.png) where $φ\_j\(x\)$ are known as basis functions

几个basis： Gaussian basis： ![](../../.gitbook/assets/Pasted%20image%2020210419211558.png) \(although it should be noted that they are not required to have a probabilistic interpretation\)

sigmoidal basis: ![](../../.gitbook/assets/Pasted%20image%2020210419211731.png) Equivalently, we can use the ‘tanh’ function because this is related to the logistic sigmoid by tanh\(a\)=2σ\(a\) − 1

\(在这章，并不具体特指某种basis，因此简记φ\(x\)= x\)

### 3.1.1 Maximum likelihood and least squares

Note that the Gaussian noise assumption implies that the conditional distribution of t given x is **unimodal**, which may be inappropriate for some applications. An extension to mixtures of conditional Gaussian distributions, which permit multimodal conditional distributions, will be discussed in Section 14.5.1.

把上边的式子改写成向量形式： ![](../../.gitbook/assets/Pasted%20image%2020210419220550.png) Note that in supervised learning problems such as regression \(and classification\), we are not seeking to model the distribution of the input variables. 因此X永远在condition这边，因此省略 ![](../../.gitbook/assets/Pasted%20image%2020210420090558.png) ![](../../.gitbook/assets/Pasted%20image%2020210420090646.png)

这时梯度为 ![](../../.gitbook/assets/Pasted%20image%2020210420090833.png) 解上面这个式子，可以得到正规方程 ![](../../.gitbook/assets/Pasted%20image%2020210420091112.png)

pseudo inverse： 当矩阵is square and invertible，pseudo inverse转化为inverse ![](../../.gitbook/assets/Pasted%20image%2020210420091243.png)

### 3.1.3 Sequential learning

if the data set is sufficiently large, it may be worthwhile to use **sequential algorithms, also known as on-line algorithms**, in which the data points are considered one at a time, and the model parameters updated after each such presentation. -&gt; stochastic gradient descent, also known as sequential gradient descent

### 3.1.4 Regularized least squares

L2对应的solution： ![](../../.gitbook/assets/Pasted%20image%2020210420103453.png)

正则化是为了约束w，因此可以写成 ![](../../.gitbook/assets/Pasted%20image%2020210420104458.png) 这样就可以用Lagrange： ![](../../.gitbook/assets/Pasted%20image%2020210420104541.png)

quadratic，约束为仿射，满足KKT，因此： ![](../../.gitbook/assets/Pasted%20image%2020210420104721.png) 可以把上面这个式子画到w各维组成的空间中，蓝色的为error的等高线，就能得到下面这张熟悉的图： ![](../../.gitbook/assets/Pasted%20image%2020210420104911.png)

### 3.1.5 Multiple outputs

## 3.2. The Bias-Variance Decomposition

h\(x\)是t的条件期望，这个在1.5.5重已经提到过 Loss的期望可以分解为以下两项，第一项y\(x\)与model有关，而第二项只取决于数据自身的noise ![](../../.gitbook/assets/Pasted%20image%2020210420192405.png)

下面来探讨model自身的uncertainty：

* 如果是bayesian方法，model的不确定性由**posterior** distribution over w决定
* 如果是frequentist方法：

    A frequentist treatment, however, involves making a point estimate of w based on the data set D, and tries instead to interpret the uncertainty of this estimate through the following thought experiment. Suppose we had a large number of data sets each of size N and each drawn independently from the distribution p\(t, x\). 

    For any given data set D, we can run our learning algorithm and obtain a prediction function y\(x; D\). Different data sets from the ensemble will give different functions and consequently different values of the squared loss. **The performance of a particular learning algorithm is then assessed by taking the average over this ensemble of data sets.**

对于上面式子的第一项进行配凑，再对于D取期望，可以得到 ![](../../.gitbook/assets/Pasted%20image%2020210420194707.png)

![](../../.gitbook/assets/Pasted%20image%2020210420195208.png)

![](../../.gitbook/assets/Pasted%20image%2020210420195312.png)

## 3.3. Bayesian Linear Regression

### 3.3.1 Parameter distribution

For the moment, we shall treat the noise precision parameter β as a known constant. 这里选择gausiian是因为，观察 ![](../../.gitbook/assets/Pasted%20image%2020210420212409.png) 可以发现p\(t\|w\) defined by \(3.10\) is the exponential of a quadratic function of w 因此共轭的先验就选择Gaussian ![](../../.gitbook/assets/Pasted%20image%2020210420212340.png)

Due to the choice of a conjugate Gaussian prior distribution, the posterior will also be Gaussian

这里使用第二章的conditional of gaussian的结论可以直接写出prior与likelihood相乘之后归一化得到的posterior的表达式 ![](../../.gitbook/assets/Pasted%20image%2020210420213133.png) ![](../../.gitbook/assets/Pasted%20image%2020210420213206.png)

因为gaussian的最值就等于均值，因此 $w\_{MAP} = w\_N$ 同时，可以观察到，当prior的$variance\to\infty$，后验的$m\_N$会转化为MLE的结果，也就是之前提到的正规方程 ![](../../.gitbook/assets/Pasted%20image%2020210420214524.png)

本章下文中，讨论的是zero-mean isotropic\(各向同性\) Gaussian governed by a single precision parameter α ![](../../.gitbook/assets/Pasted%20image%2020210420214737.png)

log of posterior: ![](../../.gitbook/assets/Pasted%20image%2020210420214840.png) 最大化log posterior等价于最小化sum-of-squares with regularization term λ = α/β

### 3.3.2 Predictive distribution

![](../../.gitbook/assets/Pasted%20image%2020210420222821.png)

利用上边的2.115的margin结论，可以直接得出output的分布： ![](../../.gitbook/assets/Pasted%20image%2020210420223100.png)

在variance的表达式中，第一项是因为数据的noise，第二项是因为w的uncertainty。而noise和w是独立的，因此是相加的关系

缺陷： If we used localized basis functions such as Gaussians, then in regions away from the basis function centres, the contribution from the second term in the predictive variance \(3.59\) will go to zero, leaving only the noise contribution β−1.

Thus, the model becomes very confident in its predictions when extrapolating outside the region occupied by the basis functions, which is generally an undesirable behaviour. This problem can be avoided by adopting an alternative Bayesian approach to regression known as a **Gaussian process**.

