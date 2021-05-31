# lecture10-MCMC-opt




MCMC所存在的问题：
1 图中，从当前位置进行random walk，很大概率会走到一个低概率的位置，也就会导致reject很多sample
![](Pasted%20image%2020210531105647.png)


2 ![](Pasted%20image%2020210531110006.png)



Random walk can have poor acceptance rate
![](Pasted%20image%2020210531110451.png)
## Using gradient information

Hamiltonian Monte Carlo

![](Pasted%20image%2020210531111010.png)
我们想建立一个关于位置x的分布，以表示其能量。但这个distribution不好写。
我们就先引入hamitonian，算出某位置的动能与势能之和，然后对于位置和速度建立一个distribution，这样分布的形式会简单些。

![](Pasted%20image%2020210531111314.png)
![](Pasted%20image%2020210531111353.png)
(我们把位置的notation从x换成q，动量的p还是p)

我们引入这个，是为了把上图中的这梯度用到MCMC的更新当中。原本下一个sample只与T(x)有关，现在我们想让T(x)和梯度共同决定，从而控制方向。

下面是一个物理中Hamiltonian的例子，随便看一下就好：
![](Pasted%20image%2020210531123743.png)
注意在我们的任务中，U(q)是true target distribution P，K(p)可以是任意一个辅助的方程，我们常用$p^2/2$. 
因此p只是一个辅助量，我们真正关注的是 $q_1,...,q_t,q_{t+1}$
而上图中一个变量的微分可以转换成另一个变量的梯度，这样我们就可以交叉更新这两个变量了。

![](Pasted%20image%2020210531125446.png)
但是上面这种方法是发散的。实际上上面的过程所对应的矩阵，行列式>1，因此在连乘之后会爆炸。

稍微改进一下之后，则可以收敛
![](Pasted%20image%2020210531125718.png)
![](Pasted%20image%2020210531125732.png)


![](Pasted%20image%2020210531125944.png)
![](Pasted%20image%2020210531130403.png)
原本我们draw x' from Q(x'|x)时，我们是在centered on x的gaussian上面随机选一个，
而现在我们是从
![](Pasted%20image%2020210531130628.png)
当中抽取，而这个梯度由上一个时刻的q决定。
![](Pasted%20image%2020210531132041.png)
注意我们在test一个sample之前，可以进行L次leapfrog

![](Pasted%20image%2020210531130833.png)
高维情况下，可以明显看到sample变得uncorrelated了
![](Pasted%20image%2020210531131004.png)
![](Pasted%20image%2020210531131102.png)

***
可见至此我们把mcmc和opt(优化理论)结合了起来。
采样的过程 - MCMC
梯度 - opt
***
变种：
只进行一次leapfrog：
![](Pasted%20image%2020210531132411.png)

***
![](Pasted%20image%2020210531133553.png)
注意HMC不能用于离散变量，因为离散变量不能求梯度
## Using approximation of the given probability distribution
***
![](Pasted%20image%2020210531133850.png)

回顾：在之前的各种方法中，哪里用到了true distribution？
MC：用P算importance
MCMC：算acceptance时
HMC：算U(q)的梯度时，U其实就是P。换一下notation的话就是$\frac{dP}{dx}$
### variational MCMC
![](Pasted%20image%2020210531144936.png)
回想之前的variational inference，之前我们只是引入λ，用q近似p(x|θ)
现在我们再给p(x|θ)增加一个variational parameter，现在p(x|θ)也是近似的了。
我们的目标就不是用q近似真实的p，而是找近似的q和近似的p能否汇合在一个比较好的结果上($p^{est}$)。

Variational MCMC:
![](Pasted%20image%2020210531150137.png)
在variational inference中，我们利用KL(Q||P)，使得Q逼近P。
而在variational MCMC中，
把刚才我们提到的收敛的位置 $p^{est}$作为proposal，从中进行下一个sample。最终我们有希望用所有sample逼近P(x).

这里是另一种MCMC结合opt的思路：
用Q和P算$p^{est}$ - variational(opt)
把$p^{est}$当作proposal进行sample - MCMC


***
#### 两种idea的比较？
variational MCMC的idea可以简化问题的structure，比如假设我们的$p^{est}$可以factorize为多个q的乘积
HMC的idea更侧重于改善具体的sample的性能

因此我们其实可以先用variational MCMC来简化问题的结构，比如把discrete转化为continuous或者施加mean field，然后再用HMC进行具体的proposal上的sample。


***
### Sequential MC
利用之前resampling的思想，然后加以推广
![](Pasted%20image%2020210531153337.png)
假设在一个HMM中，X代表latent variable，Y代表data
![](Pasted%20image%2020210531153559.png)
求$p(X_t|Y_{1:t})$的inference，其实可以改写成从$p(X_t|Y_{1:t-1})$中进行weighted sample的过程。

假设我们现在来了一个新的$Y_{t+1}$, 想求$p(X_{t+1}|Y_{t+1}, Y_{1:t})$。
我们通过sample来预测下一个时刻，隐变量X的状态。其中$p(X_t|Y_{1:t})$就是刚才sample得到的：
![](Pasted%20image%2020210531154503.png)
然后我们把新的data放进来，其实就相当于进行下一次的sample
![](Pasted%20image%2020210531154847.png)

![](Pasted%20image%2020210531155245.png)









