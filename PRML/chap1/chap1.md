## 1.1. Curve Fitting
1. As we shall see in Chapter 3, the number of parameters is not necessarily the most appropriate measure of model complexity. 参数个数不足以衡量模型的复杂度
2. We shall see that the least squares approach to finding the model parameters represents a specific case of maximum likelihood (discussed in Section 1.2.5), and that the over-fitting problem can be understood as Section 3.4 a general property of maximum likelihood 最小二乘法可以看作MLE，过拟合是MLE的property

## 1.2. Probability Theory
![](Pasted%20image%2020210324204605.png)
**Bayes’ theorem**
![](Pasted%20image%2020210324204747.png)
![](Pasted%20image%2020210324205005.png)
We can view the denominator in Bayes’ theorem as being the normalization constant required to ensure that the sum of the conditional probability on the left-hand side of (1.12) over all values of Y equals one.

***

If we had been asked which box had been chosen before being told the identity of the selected item of fruit, then the most complete information we have available is provided by the probability p(B). We call this the **prior probability** because it is the probability available before we observe the identity of the fruit. 

Once we are told that the fruit is an orange, we can then use Bayes’ theorem to compute the probability p(B|F), which we shall call the **posterior probability** because it is the probability obtained after we have observed F.

***

We note that if the joint distribution of two variables factorizes into the product of the marginals, so that p(X, Y ) = p(X)p(Y ), then X and Y are said to be **independent**. From the product rule, we see that p(Y |X) = p(Y ), and so the conditional distribution of Y given X is indeed independent of the value of X.

## 1.2.1 Probability densities
The probability that x will lie in an interval (a, b) is then given by
![](Pasted%20image%2020210324210333.png)
满足
![](Pasted%20image%2020210324210521.png)

***

下面这里提到了一个jacobian factor，大概意思就是：
概率密度函数和普通的函数是不同的。
**在pdf中，如果两个随机变量有非线性的函数关系，这两个函数的pdf不会保持这种关系。**
进而可以推广出结论，pdf的最值与variable的选择有关
![](Pasted%20image%2020210324214908.png)

**下面是对于“pdf的最值与variable的选择无关”的说明：**
由下面的推导，可知对于普通的非线性函数，最值的非线性关系是保持的：
![](Pasted%20image%2020210324215859.png)

但是
由于有这个式子：
![](Pasted%20image%2020210324220810.png)
![](Pasted%20image%2020210324220710.png)
在(4)中，因为有第二项，所以左边等于0时 $p_{x}^{\prime} (g(y))$ 不一定等于0，因此两个极值不一定同时达到。
需要注意的是，当 $x=g(y)$ 为线性变化时，$g^{\prime\prime}(y)=0$ ，因此上面式子里的第二项就没有了，此时关系保持。

![](Pasted%20image%2020210324222232.png)
上图中，从$x$ 变换到 $y$经历了一个非线性变换。如果不考虑jacobian factor，应该是红线转移到绿线，最值保持函数关系。但是因为有jacobian factor，实际上转移到了紫色的线，最值并不符合函数关系

***

cumulative distribution function：
![](Pasted%20image%2020210324222703.png)


The sum and product rules
![](Pasted%20image%2020210324222833.png)


## 1.2.2 Expectations and covariances

Expectation of f(x)：
![](Pasted%20image%2020210324223635.png)
![](Pasted%20image%2020210324223649.png)
可以用有限的$N$次sample近似求expectation：
![](Pasted%20image%2020210324223840.png)
We shall make extensive use of this result when we discuss sampling methods in Chapter 11. The approximation in (1.35) becomes **exact** in the limit $N\to\infty$.


用下标表示which variable is being averaged over，
在 $\mathbb{E}_{x}f(x,y)$ 中，是对 $x$ 取平均， $\mathbb{E}_{x}f(x,y)$ will be a function of $y$ .


***Conditional Expectation***
![](Pasted%20image%2020210324225812.png)


Variance:
![](Pasted%20image%2020210324230050.png)
![](Pasted%20image%2020210324230211.png)
![](Pasted%20image%2020210324230237.png)
![](Pasted%20image%2020210324230247.png)
If we consider the covariance of the components of a vector x with each other, then we use a slightly simpler notation $cov[\mathrm {x}]\equiv cov[\mathrm {x}, \mathrm {x}]$.
 ## 1.2.3 Bayesian probabilities
 对于不可重复的实验，如冰川会不会融化，我们不能通过频率来描述uncertainty，这时候要通过probability来描述。此时每当我们掌握了一些新的证据，都会对原有的估计加以修正，这就是bayesian的思路。
 
 prior, posterior, likelihood之间的关系，这个应该看了好多遍了：
 ![](Pasted%20image%2020210325150440.png)
 ![](Pasted%20image%2020210325150459.png)
 分母是normalization term，因为如果左右两边同时对$\mathrm{w}$积分：
 ![](Pasted%20image%2020210325150708.png)
 
 ***
 
 In a frequentist setting, $\mathrm{w}$ is considered to be a **fixed** parameter, whose value is determined by some form of ‘estimator’, and error bars on this estimate are obtained by considering the distribution of possible data sets $\mathcal{D}$. 
 By contrast, from the Bayesian viewpoint there is only a single data set $\mathcal{D}$ (namely the one that is actually observed), and the **uncertainty in the parameters is expressed through a probability distribution over $\mathrm{w}$.**
 
 ***
 
 ## 1.2.4 The Gaussian distribution
 ![](Pasted%20image%2020210325151909.png)
 The square root of the variance σ, is called the *standard deviation*
 β = 1/σ2, is called the *precision*
 ![](Pasted%20image%2020210325152145.png)
 ![](Pasted%20image%2020210325152155.png)
 ![](Pasted%20image%2020210325152218.png)
 
 ![](Pasted%20image%2020210325152313.png)
 
 
 One common criterion for determining the parameters in a probability distribution using an observed data set is to find the parameter values that maximize the likelihood function. 
 This might seem like a strange criterion because, from our foregoing discussion of probability theory, **it would seem more natural to maximize the probability of the parameters given the data, not the probability of the data given the parameters.**
 
 MLE得到的$\mu_{ML}$就是sample mean，$\sigma_{ML}$就是sample variance
 ![](Pasted%20image%2020210325170548.png)
 ![](Pasted%20image%2020210325170557.png)
 ![](Pasted%20image%2020210325170606.png)
 
 maximum likelihood approach systematically underestimates the variance of the distribution.
 ![](Pasted%20image%2020210325171017.png)
 
 ## 1.2.5 Curve fitting re-visited
 
 we shall assume that, given the value of x, the corresponding value of t has a Gaussian distribution with a mean equal to the value y(x, w) of the polynomial curve
 ![](Pasted%20image%2020210325193340.png)
 写出likelihood：
 ![](Pasted%20image%2020210325193715.png)
 求log，后两项与$\mathrm{w}$无关
 ![](Pasted%20image%2020210325193824.png)
 the sum-of-squares error function has arisen as a consequence of **maximizing likelihood under the assumption of a Gaussian noise distribution**
 
 同时，对$\beta$求导，可以得出
 ![](Pasted%20image%2020210325194125.png)
 此时就可以用这个分布进行预测了
 ![](Pasted%20image%2020210325194319.png)
 
 ***
 
 introduce a prior distribution over the polynomial coefficients $\mathrm{w}$
 ![](Pasted%20image%2020210325200842.png)
 此时likelihood：
 ![](Pasted%20image%2020210325203838.png)
 可以看到相当于加入了正则项：
 ![](Pasted%20image%2020210325203913.png)
 This technique is called **maximum posterior**, or simply **MAP**.
 
 ## 1.2.6 Bayesian curve fitting
 虽然前面求出了$\mathrm{w}$的后验，但这不能算是完成的bayesian treatment。we should consistently apply the sum and product rules of probability, which requires, as we shall see shortly, that we integrate over all values of $\mathrm{w}$。
 
We therefore wish to evaluate the predictive distribution $p(t|x, \mathbf{x}, \mathbf{t})$(这里设$\alpha$和$\beta$已知)
![](Pasted%20image%2020210325222016.png)
其中
![](Pasted%20image%2020210325222034.png)
Here p(w|x, t) is the posterior distribution，通过$\frac{prior\times likelihood}{normalization\space term}$得到，section3.3中可知，对于curve fitting，posterior也是一个gaussian

此外更进一步。预测结果也是一个gaussian：
![](Pasted%20image%2020210325223259.png)
The first term in (1.71) represents the uncertainty in the predicted value of t due to the noise on the target variables and was expressed already in the maximum likelihood predictive distribution (1.64) through ${β^{-1}_{ML}}$. However, the second term arises from the uncertainty in the parameters w and is a consequence of the Bayesian treatment.


## 1.3. Model Selection
Akaike information criterion, or AIC chooses the model for which the quantity ![](Pasted%20image%2020210325224119.png)
is largest. Here p(D|wML) is the best-fit log likelihood, and **M is the number of adjustable parameters in the model.**

section 4.4.1中要讲到 Bayesian information criterion, or BIC
Such criteria do not take account of the uncertainty in the model parameters, 
however, and in practice they tend to favour overly simple models. We therefore turn in Section 3.4 to a fully Bayesian approach where we shall see how complexity penalties arise in a natural and principled way.

## 1.5. Decision Theory
作用：Here we turn to a discussion of decision theory that, **when combined with probability theory, allows us to make optimal decisions in situations involving uncertainty** such as those encountered in pattern recognition.

