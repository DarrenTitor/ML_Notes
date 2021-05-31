![](Pasted%20image%2020210529101739.png)
![](Pasted%20image%2020210529101758.png)
![](Pasted%20image%2020210529102013.png)

![](Pasted%20image%2020210529102033.png)

![](Pasted%20image%2020210529102327.png)

![](Pasted%20image%2020210529102341.png)
![](Pasted%20image%2020210529102359.png)

![](Pasted%20image%2020210529102413.png)


![](Pasted%20image%2020210529102436.png)
![](Pasted%20image%2020210529102502.png)

![](Pasted%20image%2020210529102527.png)








# Theory of Variational Inference

variational就是把问题转化为一个优化问题的方法。
在graphical model中，inference或者说partition function往往特别难算，经常是指数级别的复杂度，因此我们尝试把partition function写成variational的形式
![](Pasted%20image%2020210529132033.png)


这个时候我们可以接触exponential family的性质，因为exponential的表达中就有一个现成的normalization term，A

![](Pasted%20image%2020210529132325.png)
其中θ要保证不会使A趋于无穷，因此θ在一个有限的集合之内
![](Pasted%20image%2020210529132556.png)

***
为什么要选exponential family？
因为μ=sufficient statistic的期望=x的margin，而margin正是我们想算的。

![](Pasted%20image%2020210529132810.png)
此外normalization term也有所对应
![](Pasted%20image%2020210529132955.png)
***

为什么不能直接对着X求期望得到μ就好了？
因为通常为指数级运算，我们想要用一个incremental的方法解决这个问题，因此接下来我们要把这个问题写成variational的形式。

![](Pasted%20image%2020210529134152.png)
注意上面这个表达式其实在PRML中已经见过：
![](Pasted%20image%2020210529134254.png)
![](Pasted%20image%2020210529134309.png)

接下来的所有工作，都是要把A(θ)写成dual of dual的形式
![](Pasted%20image%2020210529134733.png)
但是我们要知道怎么求A*(μ)，否则没法算
***
下面举一个求Bernoulli的A*(μ)的例子：
如果通过正常的inference，我们会得到下面的结果：
![](Pasted%20image%2020210529135155.png)
下面通过conjugate，求stationary condition：
![](Pasted%20image%2020210529135734.png)
但是注意这里的μ是有值域的：
然后我们就得出了conjugate的完整式子：
![](Pasted%20image%2020210529135859.png)
有了conjugate，我们就能把原本的A写成dual of dual，然后求解，最后求得的结果跟直接算inference是一样的：
![](Pasted%20image%2020210529140115.png)

***

接下来我们推广到整个exponential family：
![](Pasted%20image%2020210529140520.png)
类比刚才Bernoulli的例子，这里的μ其实是有值域的，因此我们现在就来确定一下推广之后值域怎么找。

现在假设μ在值域内，是valid的
![](Pasted%20image%2020210529195727.png)
![](Pasted%20image%2020210529195927.png)
观察这个式子，实际上是原本distribution的negative entropy。
![](Pasted%20image%2020210529200055.png)
***
至此，我们把求μ的问题转化为了求A*(μ)、进而求A(μ)的问题。
但这样也没有完全减少他的计算量，原本是积分计算量大，现在求inverse of gradient和求entropy的时候计算量也很大。
![](Pasted%20image%2020210529200618.png)

接下来关注两个问题：
1. μ什么时候有解？
2. 如何approximate μ的解空间，使得我们求解整个问题时更好算？

***
μ什么时候有解？
![](Pasted%20image%2020210529203105.png)
μ可以看作是φ的加权求和，然后概率又满足一定的约束，
事实上，μ是φ的Convex combination(凸组合指点的线性组合，要求所有系数都非负且和为1).
因此这个marginal polytope就是extreme points of sufficient statistics的Convex hull
![](Pasted%20image%2020210529203435.png)

根据这个定理，这个polytope可以写成**finite**个linear inequality的组合
![](Pasted%20image%2020210529203516.png)
看下面这个例子，
这里我们把sufficient statistics设为singleton和pairwise的拼接，可以看到最终结果是φ的极值点所构成的convex hull，而且可以进一步写成4个不等式。
![](Pasted%20image%2020210529203800.png)
![](Pasted%20image%2020210529205441.png)
但是现在的问题是，之前那个定理只保证linear inequality有finite个，然而这个个数其实是很大的。
可以看到下图中的结论，tree graphical model的约数个数是随着graph size线性增长的(这也解释了tree结构的好处)
![](Pasted%20image%2020210529204131.png)

因此下面就来看如何approximate这个polytope
***
如何approximate μ的解空间？
![](Pasted%20image%2020210529210805.png)

#### Mean field approximation
![](Pasted%20image%2020210529213639.png)
我们呢可以去掉graph中的一些边，或者所有边，然后用新的μ空间来近似真正的μ空间。

去掉所有边时，$\theta_{ij}=0$, $\mu_{ij}=P(x_i,x_j)=P(x_i)P(x_j)=\mu_i\mu_j$

从这个视角，我们可以把marginal的approximation转化为对θ的集合的近似或者对μ的集合(polytope)的近似.

![](Pasted%20image%2020210529214512.png)
可以看到，由于这时的θ只是真正的θ集合的子集(因为增加了约束)，因此对应到polytope上就变成了inner approximation。

![](Pasted%20image%2020210529215130.png)
当我们去掉了所有的边，我们就是用所有变量在Q上的margin的连乘去近似整个joint，由此我们可以直接写出A*，也就是这个近似的entropy，其实等于所有变量在Q上的margin的extropy之和。
而在原本的P上，我们是求不出来整个joint的entropy的。

![](Pasted%20image%2020210529220858.png)

#### Bethe Approximation and Sum-Product
对于任意一个graph，
![](Pasted%20image%2020210529224716.png)
注意上图中的这两个条件：singleton的margin和为1；pairwise potential在对于其中一个变量sum之后得到另一个变量的margin
我们的μ要保证local consistency，但是我们丢掉global consistency和其他的各种约束。
因此我们的集合其实是比marginal polytope大，是一个outer bound。
在求A*(μ)时，我们用tree的entropy来近似任意graph的entropy
![](Pasted%20image%2020210529225321.png)

***