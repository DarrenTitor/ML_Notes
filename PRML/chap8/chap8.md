# 8. GRAPHICAL MODELS


**probabilistic graphical models**

In a probabilistic graphical model, each node represents a random variable (or group of random variables), and the links express probabilistic relationships between these variables.

有向图：Bayesian networks
无向图：Markov random fields

有向图 expressing causal relationships between random variables
无向图 expressing soft constraints between random variables.

在solving inference problems时，一般为了计算方便，要把有向图和无向图转化为factor graph

## 8.1. Bayesian Networks

举个例子，任意一个3个随机变量的分布都可以写成这种形式，并画出bayesian network
![](Pasted%20image%2020210331200923.png)
![](Pasted%20image%2020210331200932.png)

需要注意，式子的左边是对称的，然而最终画出来的图不对称。展开的顺序不同，最终形成的图也就不一样。

扩展到K个变量：如果按这样展开，则形成的图是fully connected 
![](Pasted%20image%2020210331201204.png)

然而，it is the **absence** of links in the graph that conveys interesting information about the properties of the class of distributions that the graph represents

**Each such conditional distribution will be conditioned only on the parents of the corresponding node in the graph. **

![](Pasted%20image%2020210331201540.png)

因此，给定一个有向图，我们便能写出对应的joint distribution：
![](Pasted%20image%2020210331201828.png)
这体现了the **factorization** properties of the joint distribution for a directed graphical model

注：这里的图必须no directed cycles，也就是DAG

## 8.1.1 Example: Polynomial regression

在Polynomial regression中：
The random variables in this model are the vector of polynomial coefficients $\mathrm{w}$ and the observed data $t=(t_1,...,t_N)^T$. 
In addition, this model contains the input data $x =(x_1,...,x_N)^T$, the noise variance $σ^2$, and the hyperparameter $\alpha$ representing the precision of the Gaussian prior over w, all of which are **parameters** of the model rather than random variables.

这样一来，$t$和$\mathrm{w}$的joint distribution就可以写作：
![](Pasted%20image%2020210331203325.png)
![](Pasted%20image%2020210331203332.png)

上图中的N个t结点不方便，We introduce a graphical notation that allows such multiple nodes to be expressed more compactly, in which we draw a single representative node tn and then surround this with a box, called a **plate**, labelled with N indicating that there are N nodes of this kind.
![](Pasted%20image%2020210331203540.png)

有时我们会把参数也写道表达式中，
![](Pasted%20image%2020210331203804.png)

而这些参数不是随机变量，不能化成空心圆，random variables will be denoted by open circles, and deterministic parameters will be denoted by smaller solid circles.
![](Pasted%20image%2020210331203906.png)


example the variables {tn} from the training set in the case of polynomial curve fitting. **In a graphical model, we will denote such observed variables by shading the corresponding nodes**. Thus the graph corresponding to Figure 8.5 in which the variables {tn} are observed is shown in Figure 8.6. Note that the value of w is not observed, and so w is an example of a **latent variable,** also known as a **hidden variable**. Such variables play a crucial role in many probabilistic models and will form the focus of Chapters 9 and 12.

![](Pasted%20image%2020210331205144.png)

（在观测到某些变量之后，我们可以写出$\mathrm{w}$的后验：）
![](Pasted%20image%2020210331210806.png)

其实参数的后验不重要，重要的是要用模型做出预测。
记新来的输入为$\hat{x}$, 我们想要找到the corresponding probability distribution of $\hat{t}$ conditioned on the observed data

![](Pasted%20image%2020210331211322.png)

and the corresponding joint distribution of all of the random variables in this model, conditioned on the deterministic parameters, is then given by
![](Pasted%20image%2020210331211340.png)

![](Pasted%20image%2020210331212222.png)

## 8.1.2 Generative models

**ancestral sampling**

We shall suppose that the variables have been ordered such that there are no links from any node to any lower numbered node, in other words each node has a higher number than any of its parents. Our goal is to draw a sample $\hat{x}_1,...,\hat{x}_K$ from the joint distribution.

![](Pasted%20image%2020210331213836.png)
Note that at each stage, these parent values will always be available becauce they correspond to lower numbered nodes that have already been sampled

To obtain a sample from some marginal distribution corresponding to a subset of the variables, we simply take the sampled values for the required nodes and ignore the sampled values for the remaining nodes.
![](Pasted%20image%2020210331214236.png)

**The primary role of the latent variables is to allow a complicated distribution over the observed variables to be represented in terms of a model constructed from simpler (typically exponential family) conditional distributions.**

***
Two cases are particularly worthy of note, namely when the parent and child node each correspond to discrete variables and when they each correspond to Gaussian variables, because in these two cases the relationship can be extended hierarchically to construct arbitrarily complex directed acyclic graphs.

***
Discrete $\to$ Discrete


***
Gaussian $\to$ Gaussian





