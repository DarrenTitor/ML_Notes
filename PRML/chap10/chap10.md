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
(Mark，这点没看懂)
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

