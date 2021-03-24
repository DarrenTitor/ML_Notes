<head>
    <script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
    <script type="text/x-mathjax-config">
        MathJax.Hub.Config({
            tex2jax: {
            skipTags: ['script', 'noscript', 'style', 'textarea', 'pre'],
            inlineMath: [['$','$']]
            }
        });
    </script>
</head>

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
上图中，从$x$变换到$y$经历了一个非线性变换。如果不考虑jacobian factor，应该是红线转移到绿线，最值保持函数关系。但是因为有jacobian factor，实际上转移到了紫色的线，最值并不符合函数关系

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
We shall make extensive use of this result when we discuss sampling methods in Chapter 11. The approximation in (1.35) becomes **exact** in the limit $$N\to\infty$$ .


用下标表示which variable is being averaged over，
在 $${E}_{x}f(x,y)$$ 中，是对 $$x$$ 取平均， $${\mathbb{E}}_{x}[f(x,y)]$$ will be a function of $$y$$ .


***Conditional Expectation***
![](Pasted%20image%2020210324225812.png)


Variance:
![](Pasted%20image%2020210324230050.png)
![](Pasted%20image%2020210324230211.png)
![](Pasted%20image%2020210324230237.png)
![](Pasted%20image%2020210324230247.png)
If we consider the covariance of the components of a vector x with each other, then we use a slightly simpler notation $cov[\mathrm {x}]\equiv cov[\mathrm {x}, \mathrm {x}]$.
