# 10. Approximate Inference

motivation:
我们想要求posterior distribution p(Z|X) of the latent variables Z given the observed (visible) data variables X
但是，
对于连续变量，the required integrations may not have closed-form analytical solutions, while the dimensionality of the space and the complexity of the integrand may prohibit numerical integration
对于离散变量，the marginalizations involve summing over all possible configurations of the hidden variables. There may be exponentially many hidden states so that exact calculation is prohibitively expensive

因此我们需要approximation schemes，而这分为两类：stochastic or deterministic approximations
**Stochastic** techniques such as Markov chain Monte Carlo, 推广了Bayesian methods的使用。Bayesian methods在有无限的运算资源时能给出精确解。而这些近似方法可以在有限的时间内给出解。In practice, sampling methods can be **computationally demanding**, often limiting their use to small-scale problems. Also, it can be **difficult to know whether a sampling scheme is generating independent samples** from the required distribution.
这章讲的是**deterministic** approximation schemes, some of which scale well to large applications. 但它们不能给出精确解

## 10.1. Variational Inference
a **functional** is a mapping that takes a function as the input and that returns the value of the functional as the output.
一个例子就是entropy $H[p]$，接受p(x)作为input，返回一个量作为输出
![](Pasted%20image%2020210502091838.png)
a functional derivative expresses how the value of the functional changes in response to infinitesimal changes to the input function

variational methods本身没有什么近似的元素。但是通常会在求解的过程中限制function的种类和范围, 比如只考虑quadratic functions或者只考虑某些fixed basis functions的线性组合. 在probabilistic inference中，可能会有factorization assumptions

***

variational optimization用于inference的例子：
我们有一个fully Bayesian model，所有latent variable记作Z，所有observed variables记作X，X由N个i.i.d数据组成。
我们设了joint distribution p(X, Z), 目标是得到近似的posterior p(Z|X)和近似的evidence p(X)

仿照之前EM的做法，我们从log marginal中拆出来KL散度，唯一与之前不同的是，我们这里没有出现theta，因为在EM中，theta都是observed的（可以去看GMM对应的graph），而在这里，the parameters are now stochastic variables and are absorbed into Z。
![](Pasted%20image%2020210502152711.png)

和之前的Estep一样，我们关于q对L进行maximize（顺便一提因为没有θ，所以没有M-step了）。如果我们对于q(Z)没有限制，选啥都行的话，lower bound达到最大时，KL散度为0达到, q(Z)就等于真正的posteriorp(Z|X). 这样的结果虽然是对的，但是可能会导致p(Z|X)太复杂，没法算。

**We therefore consider instead a restricted family of distributions q(Z) and then seek the member of this family for which the KL divergence is minimized.** 我们想让对p(Z)加一些限制，使得它们都“好算”，同时又能对真正的posterior足够近似
(下面这段，因为模型简化，所以泛化能力变强，因此就解决了过拟合的问题)
In particular, there is no ‘over-fitting’ associated with highly flexible distributions. Using more flexible approximations simply allows us to approach the true posterior distribution more closely.

其中一种约束q的方法是使用parametric distribution，如果q(Z|ω) governed by a set of parameters ω，那么L就成了ω的function。我们就可以把常用的优化方法用到L上面了

### 10.1.1 Factorized distributions
现在介绍另一种限制q(Z)的方法：
这里的Z其实是$\mathrm{Z}$, 是所有的latent variable，这里就简写了
Suppose we partition the elements of Z into disjoint groups that we denote by $Z_i$ where i =1,...,M. We then assume that the q distribution factorizes with respect to these groups, so that
![](Pasted%20image%2020210502162556.png)

注意我们对于$q_i(Z)$的具体形式没有进一步的假设，就只是假设q(Z)能拆开而已
This factorized form of variational inference corresponds to an approximation framework developed in physics called **mean field theory**

在所有“能拆”的q(Z)中，我们要找能使L(也就是lower bound)最大的。我们可以分别对每一个$q_i(Z)$都轮流优化，也就是优化某一个，固定其他的，然后轮流进行。

把$q_i(Z)$连乘代入到L的定义式中，可以得到
![](Pasted%20image%2020210502165955.png)
这里定义了一个新的distribution$\hat{p}(X,Z_j)$：
![](Pasted%20image%2020210502170219.png)
然后又定义了一个$E_{i\neq j}[...]$，代表这个期望在算的过程中把j这项去掉了。注意下面这个式子算lnp(X,Z)的“期望”，是对于Z而言的，因此乘的是Z的分布q
![](Pasted%20image%2020210502170435.png)

而具体固定住其他的、只优化j的过程也有简便计算，我们想maximize L，可以看到L的这个形式：
![](Pasted%20image%2020210502170959.png)
就是一个negative Kullback-Leibler divergence between $q_j(Z_j)$ and $\hat{p}(X,Z_j)$，因此$q_j(Z_j)=\hat{p}(X,Z_j)$的时候KL散度最小为0，L最大

此时我们得到一个optimal solution$q^*_j(Z_j)$的通用结论：
![](Pasted%20image%2020210502171340.png)
Note:
观察一下这个式子，It says that **the log of the optimal solution for factor $q_j$ is obtained simply by considering the log of the joint distribution over all hidden and visible variables and then taking the expectation with respect to all of the other factors $\{q_i\}\space for\space i\neq j$.**
而上面的这个常数项是normalization term，通常不会硬算这个const，而是先算前面这项，然后观察出const
![](Pasted%20image%2020210502174430.png)

基本流程是先适当地初始化各个$q_i(Z_i)$，然后开始循环，依次更新$q_i(Z_i)$
Convergence is guaranteed because bound is convex with respect to each of the factors $q_i(Z_i)$

### 10.1.2 Properties of factorized approximations
(Let us consider for a moment the problem of approximating a general distribution by a factorized distribution)

#### factorized Gaussian
假设有
![](Pasted%20image%2020210502181008.png)
two correlated variables z =(z1,z2) in which the mean and precision have elements
![](Pasted%20image%2020210502181108.png)
并且因为是对称矩阵，$\Lambda_{12}=\Lambda_{21}$


现在我们想要用q(z)= q1(z1)q2(z2)近似p(z)
想要求$q^*_j(Z_j)$，直接代入之前的通用表达式，注意E中只留下与$z_1$有关的部分，其他的扔到const里
![](Pasted%20image%2020210502223930.png)

然后我们发现$lnq^*_j(Z_j)$是quadratic的，因此$q^*_j(Z_j)$可以看作是一个gaussian。
注意：我们没有假设q(Z)是Gaussian，but rather we derived this result by variational optimization of the KL divergence over all possible distributions q(zi).

用配方法，可以得到$q_1^*$和$q_2^*$对应的Gaussian的参数：
![](Pasted%20image%2020210502224821.png)

Note that these solutions are **coupled**, so that q*(z1) depends on expectations computed with respect to q*(z2) and vice versa. 

通常我们会循环更新这两个，直到他们收敛。
但是在上面这个例子中，问题很简单，可以直接观察出close form solution：


In particular, because $E[z1]= m1$ and $E[z2]= m2$, we see that the two equations are satisfied if we take $E[z1]= µ1$ and $E[z2]= µ2$, and it is easily shown that this is the only solution provided the distribution is nonsingular.

![](Pasted%20image%2020210503112527.png)
可以看到我们近似出了正确的μ，但是variance过小了。
**通常来说factorized variational approximation会给出一个过于compact的distribution**

***
相反，如果我们minimize **reverse** Kullback-Leibler divergence KL(p||q)呢？
这种方法用于另一种近似的framework：expectation propagation

此时KL散度可以写成：
![](Pasted%20image%2020210503113047.png)
此时我们可以把$i\neq j$的部分当作常数，然后因为有q_j积分为1，因此有约束，要用lagrange
![](Pasted%20image%2020210503113441.png)
![](Pasted%20image%2020210503113719.png)

我们发现，qj(Zj)的最优解只与p(Z)有关，而且是close form，不需要iteration

![](Pasted%20image%2020210503114004.png)
We see that once again the mean of the approximation is correct, but that it places significant probability mass in regions of variable space that have very low probability.

***
分析这两种方法不同的原因：

对于KL(q||p)：
here is a large positive contribution to the Kullback-Leibler divergence
![](Pasted%20image%2020210503114711.png)
from regions of Z space in which p(Z) is near zero **unless** q(Z) is also close to zero.
换句话说：首先注意到ln函数越靠近0，越陡。当分子p接近0，分母q不接近0时，整个ln就很小，KL散度就很大。而我们minimize KL散度，就遏制这种情况发生，也就是说：q要缩在p的内部，对应上面的图a
对于KL(p||q)：
就正相反，q要缩到p的内部，对应图b

![](Pasted%20image%2020210503142122.png)
对于一个multimodal的分布，我们minimize KL散度可以得到一个unimodal的近似分布，而此时选用这两种KL散度就会有不同的结果。
与之前我们得到的结论一致，KL(p||q)会得到a，包在真实分布p的外面，KL(q||p)会得到b或c，缩在真实分布之内。
而KL(p||q)得到的这个分布通常表现不是很好，因为because the average of two good parameter values is typically itself not a good parameter value
不过KL(p||q)会在10.7讨论expectation propagation时发挥作用。

***

**alpha family** of divergences

![](Pasted%20image%2020210503232856.png)

$\alpha\to 1$对应KL(p||q)
$\alpha\to -1$对应KL(q||p)
对于所有的$\alpha$都有$D_\alpha(p||q)>0$ if and only if p(x)=q(x)
For α<=−1 the divergence is **zero forcing**, so that any values of x for which p(x)=0 will have q(x)=0, and typically q(x) will under-estimate the support of p(x) and will tend to seek the mode with the largest mass.
Conversely for α>=1 the divergence is zero-avoiding, so that values of x for which p(x) > 0 will have q(x) > 0, and typically q(x) will stretch to cover all of p(x), and will over-estimate the support of p(x).
α =0时的情况，we obtain a symmetric divergence that is linearly related to the **Hellinger distance** given by
![](Pasted%20image%2020210503235725.png)

### 10.1.3 Example: The univariate Gaussian


## 10.2. Illustration: Variational Mixture of Gaussians


首先根据上一章的讨论，我们可以得到下面这两个式子：
（这里只不过把N个样本合到了一起）
![](Pasted%20image%2020210513092752.png)
![](Pasted%20image%2020210513092802.png)
注意这里用precision matrices rather than covariance matrices来简化表达

然后我们给$\pi$设一个先验，因为Z的分布是multinomial的，因此我们用dirichlet
![](Pasted%20image%2020210513095641.png)
where by symmetry we have **chosen the same parameter α0 for each of the components**, and C(α0) is the normalization constant for the Dirichlet distribution
![](Pasted%20image%2020210513095812.png)
As we have seen, the parameter α0 can be interpreted as the effective
prior number of observations associated with each component of the mixture. 
If the value of α0 is small, then the posterior distribution will be influenced primarily by the data rather than by the prior.

Similarly, we introduce an independent Gaussian-Wishart prior governing the mean and precision of each Gaussian component, given by
![](Pasted%20image%2020210513101652.png)
because this represents the conjugate prior distribution when both the mean and precision are unknown
![](Pasted%20image%2020210513102257.png)
这里我们可以看到，latent variable和parameter最大的区别就是latent variable在plate中，个数随着数据增长而增长。而在graph层面，其实与parameter没有根本上的区别。

### 10.2.1 Variational distribution
总的joint可以写作
![](Pasted%20image%2020210513103038.png)
Note that only the variables X are observed.

然后我们现在就来设一个可以factorize的variational distribution
![](Pasted%20image%2020210513122043.png)
这里我们factorize between the latent
variables and the parameters 

It is remarkable that this is the **only** assumption that we need to make in order to obtain a tractable practical solution to our Bayesian mixture model.
有了这个factorize的假设，q的形式就能自动定下来
#### q(Z)
接下来，我们直接套用之前的最优解的结论，可以得到：
![](Pasted%20image%2020210513123337.png)
然后我们只保留Z相关的部分，其他的放到const中
![](Pasted%20image%2020210513123417.png)

代入之前的两个p的定义，得到
![](Pasted%20image%2020210513123500.png)
where D is the dimensionality of the data variable x.

同时取对数
![](Pasted%20image%2020210513123543.png)
然后再normalize一下，可以得到：
![](Pasted%20image%2020210513123613.png)
![](Pasted%20image%2020210513123622.png)

We see that the optimal solution for the factor q(Z) takes the same functional form as the prior p(Z|π).
然后我们可以得到期望：（类比multinomial的期望）
![](Pasted%20image%2020210513124850.png)
这个$r_{nk}$其实就起着responsibility的作用

观察到q*(Z)的最优解中有其他变量的期望，因此我们还是要用iterative的方法求解

#### q(π, µ, Λ)
下面定义三个statistics，用于简化表达：
![](Pasted%20image%2020210513125858.png)
Note that these are analogous to quantities evaluated in the maximum likelihood EM algorithm for the Gaussian mixture model

首先代入最优解：
![](Pasted%20image%2020210513131040.png)
（这里观察到右边的项要么只有π，要么只有µ和Λ。体现了q(π, µ, Λ)在这里分解为q(π)q(µ, Λ)）
而且我们观察到含有µ和Λ的项都有求和，因此我们可以把整个factorization写成：
![](Pasted%20image%2020210513131351.png)

先求关于π的：
把与π无关的都丢到const中，可以得到
![](Pasted%20image%2020210513131718.png)
Taking the exponential of both sides, we recognize q*(π) as a Dirichlet distribution
![](Pasted%20image%2020210513131821.png)

再求关于µ和Λ的：
the variational posterior distribution $q*(µk, Λk)$ does not factorize into the product of the marginals, but we can always use the product rule to write it in the form $q*(µk, Λk)= q*(µk|Λk)q*(Λk)$ .

结果也是一个Gaussian-Wishart distribution：(这里推导就跳了)
![](Pasted%20image%2020210513134027.png)
These update equations are analogous to the M-step equations of the EM algorithm for the maximum likelihood solution of the mixture of Gaussians. 
We see that the computations that **must be performed in order** to update the variational posterior distribution over the model parameters 

然而问题还没解决，
上面的最优解中需要用到$r_{nk}$, $r_{nk}$是由$\rho _{nk}$normalize得到的，$\rho _{nk}$中又要用到$E[ln\pi_k]$,$E[ln\Lambda_k]$和
![](Pasted%20image%2020210513213456.png)
![](Pasted%20image%2020210513212719.png)
We see that this expression involves expectations with respect to the variational distributions of the parameters, and these are easily evaluated to give
![](Pasted%20image%2020210513213605.png)
![](Pasted%20image%2020210513213809.png)
![](Pasted%20image%2020210513213654.png)

***
variational EM中，
![](Pasted%20image%2020210513214006.png)
MLE EM中，
![](Pasted%20image%2020210513214023.png)
可以看到形式是很相近的

***
总结：
* In the variational equivalent of the E step,
	we use the current distributions over the model parameters to evaluate the moments in (10.64), (10.65), and (10.66) and hence evaluate E[znk]= rnk.
* in the subsequent variational equivalent of the M step，
	we **keep these responsibilities fixed** and use them to **re-compute** the variational distribution over the parameters using (10.57) and (10.59).
	
同时我们可以观察到，我们求得的variational posterior q与我们的假设p的函数形式是一样的（dirichlet、Gaussian-Wishart）。This is a general result and is a consequence of the choice of conjugate distributions.

***
![](Pasted%20image%2020210513221703.png)
**Components that take essentially no responsibility for explaining the data points have rnk$\to$ 0 and hence Nk$\to$ 0. From (10.58), we see that αk$\to$ α0 and from (10.60)–(10.63) we see that the other parameters revert to their prior values**

***
In fact if we consider the limit N →∞then the Bayesian treatment converges to the maximum likelihood EM algorithm.

计算量大的部分主要是the evaluation of the responsibilities, together with the evaluation and inversion of the weighted data covariance matrices，而这些在MLE EM中也都有。因此variational的方法并没有增大多少计算量。
优点：
* 解决了singlarity的问题，these singularities are removed if we simply introduce a prior and then use a MAP estimate instead of maximum likelihood
* there is no over-fitting if we choose a large number K of components in the mixture
* the variational treatment opens up the possibility of determining the optimal number of components in the mixture without resorting to techniques such as cross validation


### 10.2.2 Variational lower bound
在求解的过程中，我们可以算一下(10.3)给出的lower bound，用于验证程序是否正确或者是否收敛。
variational GMM的lower bound：
![](Pasted%20image%2020210513233250.png)
![](Pasted%20image%2020210513235518.png)
![](Pasted%20image%2020210513235550.png)
![](Pasted%20image%2020210513235625.png)
![](Pasted%20image%2020210513235637.png)
![](Pasted%20image%2020210513235657.png)
Note that the terms involving expectations of the logs of the q distributions simply represent the negative entropies of those distributions


### 10.2.3 Predictive density
在预测中，我们要利用X建模，然后为了一个新来的$\hat{x}$找到它的$\hat{z}$
![](Pasted%20image%2020210514204156.png)
where p(π, µ, Λ|X) is the (unknown) true posterior distribution of the parameters

然后对$\hat{z}$进行summation：
![](Pasted%20image%2020210514205510.png)

然后接下来的积分就没法算了，然后用variational approximation q(π)q(µ, Λ)来代替p(π, µ, Λ|X)

![](Pasted%20image%2020210514205644.png)


### 10.2.4 Determining the number of components
还有一个需要强调的地方，
For any given setting of the parameters in a Gaussian mixture model , there will exist other parameter settings for which the density over the observed variables will be identical.