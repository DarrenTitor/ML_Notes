In a frequentist treatment, we choose specific values for the parameters by optimizing some criterion, such as the likelihood function. 
By contrast, in a Bayesian treatment we introduce prior distributions over the parameters and then use Bayes’ theorem to compute the corresponding posterior distribution given the observed data.

## 2.1. Binary Variables

Bernoulli Distribution:

![](Pasted%20image%2020210329214620.png)

当我们有N个数据，独立同分布iid，就可以写出likelihood：
![](Pasted%20image%2020210329214934.png)
![](Pasted%20image%2020210329214943.png)

对于frequentist视角，自然要求解μ使得likelihood最大：
![](Pasted%20image%2020210329215137.png)
但是这里有个需要注意的点：
在log likelihood中，μ和1-μ都是常量，结果就是likelihood只取决于$\Sigma_{n}x_{n}$有关。知道这个值，就可以求得likelihood
$\Sigma_{n}x_{n}$这个值，我们称为sufficient statistic
对μ求导，可以得到
![](Pasted%20image%2020210330134016.png)
也就是sample mean。
记x=1有m个样本，则
![](Pasted%20image%2020210330134151.png)


Binomial distribution
distribution of the number m of observations of x =1,
given that the data set has size N
![](Pasted%20image%2020210330134527.png)

对于独立事件，the mean of the sum is the sum of the means, and the variance of the sum is the sum of the variances
因此我们可以得出期望和方差：
![](Pasted%20image%2020210330134634.png)


## 2.1.1 The beta distribution

**conjugacy**
If we choose a prior to be proportional to powers of µ and (1 − µ), then the posterior distribution, which is proportional to the product of the prior and the likelihood function, will have the same functional form as the prior.

beta distribution
![](Pasted%20image%2020210330143537.png)
![](Pasted%20image%2020210330143625.png)
由分部积分法：
![](Pasted%20image%2020210330145450.png)
![](Pasted%20image%2020210330150033.png)

The posterior distribution of µ is now obtained by multiplying the beta prior(2.13) by the binomial likelihood function (2.9) and normalizing.
![](Pasted%20image%2020210330151026.png)
where l = N − m

对比(2.13)的形式，我们可以发现posterior也是一个Beta分布：
![](Pasted%20image%2020210330151321.png)


**We see that the effect of observing a data set of m observations of x =1 and l observations of x =0 has been to increase the value of a by m, and the value of b by l, in going from the prior distribution to the posterior distribution**
我们可以把观测数据的过程看作是一个“不断修改Beta分布的参数的过程”，由此从先验转化为后验

This allows us to provide a simple interpretation of the hyperparameters a and b **in the prior** as an **effective number of observations** of x =1 and x =0, respectively.
这句话很难理解，我的理解是：观测数据的过程是一个不断增加a和b的过程，也是一个不断修正先验的过程。
而在这个过程之前，当我们观测0个数据时，我们对先验的假设也可以看作观察了某几个数据的结果（fictitious prior observation）。而总体的值就叫做effective number of observations。也就是截止到目前为止，我们观测到的数据个数（包括我们最初“蒙对的”）

预测时：
![](Pasted%20image%2020210330154351.png)
由于之前提到的beta distribution的mean的公式，
![](Pasted%20image%2020210330154605.png)

which has a simple interpretation as the total fraction of observations (**both real observations and fictitious prior observations**) that correspond to x =1

由此可以得出MLE与MAP的关系：
当m, l趋于无穷时，MAP等于MLE
而对于有限的数据，the posterior mean for µ always lies between the prior mean and the maximum likelihood estimate for µ corresponding to the relative frequencies of events given by (2.7).


***

![](Pasted%20image%2020210330160406.png)
我们可以看到，在Beta distribution中，随着观察数的增加，分布变得越来越sharply peaked。
In fact, we might wonder **whether it is a general property** of Bayesian learning that, as we observe more and more data, the uncertainty represented by the posterior distribution will steadily decrease.
那么这是不是一个共有的性质呢
on average, the posterior variance of θ is smaller than the prior variance


## 2.2. Multinomial Variables

**a generalization of the Bernoulli distribution**

如果x是one hot vector，且the probability of $x_{k} =1$ is $\mu_{k}$
![](Pasted%20image%2020210330170942.png)

其中$\Sigma_{k}\space\mu_{k}=1$且$\mu_{k}\ge0$

![](Pasted%20image%2020210330171717.png)

随后可以写出likelihood的形式：
![](Pasted%20image%2020210331144200.png)

类比于binomial，这里的likelihood也只与k个sum有关：
![](Pasted%20image%2020210331144241.png)
也是sufficient statistics

由拉格朗日可以求出$\mu_{ML}$,也就是MLE的解
![](Pasted%20image%2020210331144441.png)

![](Pasted%20image%2020210331144614.png)


**multinomial distribution**

joint distribution of the quantities m1,...,mK, conditioned on the parameters µ and on the total number N of observations


![](Pasted%20image%2020210331145031.png)
![](Pasted%20image%2020210331145038.png)

### 2.2.1 The Dirichlet distribution
![](Pasted%20image%2020210331170110.png)
![](Pasted%20image%2020210331170120.png)

后验的形式：
![](Pasted%20image%2020210331170207.png)

可见后验也是一个Dirichlet
由此通过观察，我们可以得到后验normalization term的参数：
![](Pasted%20image%2020210331170336.png)
其中where we have denoted m =(m1,...,mK)T.


Beta与Dirichlet的关系：
As for the case of the binomial distribution with its beta prior, we can interpret the parameters $\alpha_{k}$ of the Dirichlet prior as an effective number of observations of $x_{k}=1$.


## 2.3. The Gaussian Distribution

![](Pasted%20image%2020210331170733.png)


***

geometrical form

gaussian中与$\mathrm{x}$有关的只有
![](Pasted%20image%2020210331172407.png)
位于指数部分。这一部分称为μ与x的Mahalanobis distance。
当Σ为identity matrix时，reduce to Euclidean distance。


