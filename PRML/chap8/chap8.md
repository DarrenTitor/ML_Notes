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

有时我们会把参数也写到表达式中，
![](Pasted%20image%2020210331203804.png)

而这些参数不是随机变量，不能画成空心圆，random variables will be denoted by open circles, and deterministic parameters will be denoted by smaller solid circles.
![](Pasted%20image%2020210331203906.png)


**In a graphical model, we will denote such observed variables by shading the corresponding nodes**. Thus the graph corresponding to Figure 8.5 in which the variables {tn} are observed is shown in Figure 8.6. Note that the value of w is not observed, and so w is an example of a **latent variable,** also known as a **hidden variable**. Such variables play a crucial role in many probabilistic models and will form the focus of Chapters 9 and 12.

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



### 8.1.3 Discrete variables

单个离散变量的概率分布是这样的
![](Pasted%20image%2020210422134343.png)
两个离散变量的分布：（$K^2-1$个参数）
![](Pasted%20image%2020210422134718.png)

![](Pasted%20image%2020210422134856.png)

From a graphical perspective, we have reduced the number of parameters by dropping links in the graph, at the expense of having a restricted class of distributions.

An alternative way to reduce the number of independent parameters in a model is by sharing parameters (also known as tying of parameters).

Another way of controlling the exponential growth in the number of parameters in models of discrete variables is **to use parameterized models for the conditional distributions** instead of complete tables of conditional probability values.
![](Pasted%20image%2020210422145217.png)
![](Pasted%20image%2020210422145040.png)
The motivation for the logistic sigmoid representation was discussed in Section 4.2.

### 8.1.4 Linear-Gaussian models

Here we show how a multivariate Gaussian can be expressed as a directed graph corresponding to a linear-Gaussian model over the component variables. 

这里要用到第二章的知识："if two sets of variables are jointly Gaussian, then the conditional distribution of one set conditioned on the other is again Gaussian"

***

这里是反过来用的，如果一个节点是高斯，而它的均值是父节点的线性组合，那么它和父节点的 joint distribution $p(\mathrm{x})$ is a multivariate Gaussian.

![](Pasted%20image%2020210422153148.png)

![](Pasted%20image%2020210422155145.png)
![](Pasted%20image%2020210422155232.png)

考虑两个极端情况：

当graph中没有连接时，there are no parameters $w_{ij}$ and so there are just D parameters $b_i$ and D parameters $v_i$
此时mean of p(x) is given by (b1,...,bD)T and the covariance matrix is diagonal of the form diag(v1,...,vD).
结果就是a set of D independent univariate Gaussian distributions

当graph为全连接时，w的矩阵是一个下三角矩阵，参数个数为D(D−1)/2.
v所在的协方差矩阵is a general symmetric covariance matrix

***
在下面这个例子中，运用上面的(8.15)和(8.16)，可以写出joint distribution的分布为：
![](Pasted%20image%2020210422165831.png)
![](Pasted%20image%2020210422165841.png)

(2.3.6)(Mark, 这段没看)
Note that we have already encountered a specific example of the linear-Gaussian relationship when we saw that the conjugate prior for the mean µ of a Gaussian
variable x is itself a Gaussian distribution over µ. The joint distribution over x and µ is therefore Gaussian. This corresponds to a simple two-node graph in which the node representing µ is the parent of the node representing x. The mean of the distribution over µ is a parameter controlling a prior, and so it can be viewed as a hyperparameter. Because the value of this hyperparameter may itself be unknown, we can again treat it from a Bayesian perspective by introducing a prior over the hyperparameter, sometimes called a hyperprior, which is again given by a Gaussian distribution. This type of construction can be extended in principle to any level and is an illustration of a hierarchical Bayesian model, of which we shall encounter further examples in later chapters.

## 8.2. Conditional Independence
对于变量a, b, c, 如果有
![](Pasted%20image%2020210422214122.png)
就称a is conditionally independent of b given c

当表示joint distribution时，表达式稍有不同，带入上面的式子就行：
![](Pasted%20image%2020210422214439.png)
（本质其实是p(a, b)=p(a)p(b)，只不过多了个condition）

conditional independence 要求上面这两个式子对变量的任意取值都要成立，而不是某些值

可以简写为：
![](Pasted%20image%2020210422214816.png)



conditional independence可以从graph中直接读出，不需要任何计算
The general framework for achieving this is called d-separation, where the ‘d’ stands for ‘directed’

### 8.2.1 Three example graphs

#### case_1
![](Pasted%20image%2020210422215337.png)
对于上面的gragh，可以写出下面的分布：
![](Pasted%20image%2020210422215532.png)
* 如果没有变量被观测，为了求p(a,b)，我们会marginalizing both sides of (8.23) with respect to c to give
![](Pasted%20image%2020210422215618.png)
这个得不出p(a)p(b)，因此得不出conditional independence，可以记作
![](Pasted%20image%2020210422215758.png)


* *假设c被观测了，或者说we condition on the variable c**，
由于本身graph都可以写成
![](Pasted%20image%2020210422220311.png)
为了求a, b的joint，我们可以直接把上边这个式子同除p(c)
![](Pasted%20image%2020210422220012.png)

(换句话说，从graph中得不到p(a,b)=p(a)p(b)，但是有p(a,b|c)=p(a|c)p(b|c))

总结：
The node c is said to be **tail-to-tail** with respect to this path because the node is connected to the tails of the two arrows,
**when we condition on node c, the conditioned node ‘blocks’ the path from a to b and causes a and b to become (conditionally) independent.**
(这里tail-to-tail的这个to就很迷惑，最好记成"and"，tail&tail, head&head, head&tail)

#### case_2
![](Pasted%20image%2020210422221418.png)
同样，先写出
![](Pasted%20image%2020210422221848.png)

* 当没有condition c时，a，b没有独立
![](Pasted%20image%2020210422221959.png)
![](Pasted%20image%2020210422222014.png)

* condition c之后，
![](Pasted%20image%2020210422222109.png)

总结：
**head-to-tail**
Such a path connects nodes a and b and renders them dependent. **If we now observe c, then this observation ‘blocks’ the path from a to b and so we obtain the conditional independence property**


#### case_3
![](Pasted%20image%2020210422231156.png)
This graph has rather different properties from the two previous examples

先写出
![](Pasted%20image%2020210422231323.png)

* Consider first the case where none of the variables are observed. Marginalizing both sides of (8.28) over c we obtain
![](Pasted%20image%2020210422231422.png)
and so **a and b are independent with no variables observed**, in contrast to the two previous examples
观测前独立
![](Pasted%20image%2020210422231534.png)

* Now suppose we condition on c,观测之后不独立
![](Pasted%20image%2020210422231738.png)

总结：
**head-to-head**
When node c is **unobserved**, it ‘blocks’ the path, and the variables a and b are **independent**. However, **conditioning** on c ‘unblocks’ the path and renders a and b **dependent**.
另外还有一个结论：**a head-to-head path will become unblocked if either the node, or any ofits descendants, is observed**，也就是说，观测c和观测c的后代可以起到同样的作用

### 8.2.2 D-separation

A, B, C是三个node的集合，没有交集
![](Pasted%20image%2020210424142519.png)

note：
parameters用small filled circles表示，理论上相当于observed nodes. However, there are no marginal distributions associated with such nodes.** Consequently they play no role in d-separation.**

We can view a graphical model (in this case a directed graph) as a filter in which a probability distribution p(x) is allowed through the filter if, and only if, it satisfies the directed factorization property (8.5)
![](Pasted%20image%2020210424145743.png)
We can alternatively use the graph to filter distributions according to whether they respect all of the conditional independencies implied by the d-separation properties of the graph.
根据d-separation theorem，上面两种方法得到的结果是一样的

#### **Markov blanket** or **Markov boundary**
The set of nodes comprising the parents, the children and the co-parents is called the **Markov blanket** 
![](Pasted%20image%2020210424151340.png)

We can think of the Markov blanket of a node xi as being the minimal set of nodes that isolates xi from the rest of the graph.

the conditional distribution of xi, conditioned on all the remaining variables in the graph, is dependent only on the variables in the Markov blanket.


## 8.3. Markov Random Fields

### 8.3.1 Conditional independence properties

We might ask whether it is possible to define an alternative graphical semantics for probability distributions such that conditional independence is determined by simple graph separation. This is indeed the case and corresponds to undirected graphical models. 也就是说undirected graph不需要复杂的separation，只需要在graph上直接分割

**If all such paths pass through one or more nodes in set C, then all such paths are ‘blocked’ and so the conditional independence property holds. **
However, if there is at least one such path that is not blocked, then the property does not necessarily hold, or more precisely there will exist at least some distributions corresponding to the graph that do not satisfy this conditional independence relation. 

![](Pasted%20image%2020210424152954.png)

**The Markov blanket for an undirected graph** takes a particularly simple form, because a node will be conditionally independent of all other nodes conditioned **only on the neighbouring nodes**
![](Pasted%20image%2020210424153232.png)

### 8.3.2 Factorization properties
我们想expressing the joint distribution p(x) as a product of functions defined over sets of variables that are local to the graph.

If we consider two nodes xi and xj that are not connected by a link, then these variables must be conditionally independent given all other nodes in the graph.
也就是说相邻节点一定相关，也就拆不开了，写不成乘积形式
![](Pasted%20image%2020210424154351.png)

clique：the set of nodes in a clique is fully connected


![](Pasted%20image%2020210424160241.png)
这里提到了两个函数，potential functions和partition function，partition function只是用于normalization的

** Note that we do not restrict the choice of potential functions to those that have a specific probabilistic interpretation as marginal or conditional distributions. **
与有向图不同，有向图中，p(x)拆成许多条件概率的乘积
在无向图中，不要求potential functions必须是marginal or conditional distributions

The presence of this normalization constant is one of the major limitations of undirected graphs.
If we have a model with M discrete nodes each having K states, 则partition function是$K^M$项求和

如果我们想要得到connection between conditional independence and factorization for undirected graphs，就要约束$\Psi_C(x_C)$ are strictly positive

由于potential functions严格大于0，因此可以写成指数的形式
![](Pasted%20image%2020210424162826.png)
where E(xC) is called an **energy function**, and the exponential representation is called the **Boltzmann distribution**
The joint distribution是potential的乘积，so the total energy is obtained by adding the energies of each of the maximal cliques.

### 8.3.4 Relation to directed graphs
#### directed -> undirected
先考虑最简单的directed chain，在chain中，maximal cliques are simply the pairs of neighbouring nodes
让这两个式子对应
![](Pasted%20image%2020210426195618.png)
![](Pasted%20image%2020210426195625.png)
就有
![](Pasted%20image%2020210426195654.png)

下面考虑general的情况：
要想转化，就要用clique的potential表示conditional distributions。
In order for this to be valid, **we must ensure that the set of variables that appears in each of the conditional distributions is a member of at least one clique of the undirected graph** 也就是说，要想达成这个，有向图中的每个节点，都要至少出现在某一个minimal clique中，不然就会把这个节点漏掉

如果有向图中的子节点只有一个parent，那么只需要把箭头直接变成edge，然后把node pair视为minimal clique
但是当一个子节点有多个parent时，也就是"head to head"时，不能直接转化。To ensure this, we add extra links between all pairs of parents of the node
Anachronistically, this process of ‘marrying the parents’ has become known as **moralization**,
and the resulting undirected graph, after dropping the arrows, is called the **moral graph**
需要注意的是，在moralization的过程中，哟可能会丢到conditiona independence，比如下面这个例子：
![](Pasted%20image%2020210426201511.png)

Steps:
1. add additional undirected links between all pairs of parents for each node in the graph and then drop the arrows on the original links to give the moral graph
2. initialize all of the clique potentials of the moral graph to 1
3. take each conditional distribution factor in the original directed graph and multiply it into one of the clique potentials
4. in all cases the partition function is given by Z=1



#### undirected -> directed
Converting from an undirected to a directed representation is much less common and in general presents problems due to the normalization constraints

#### difference
It turns out that the two types of graph can **express different conditional independence properties**


## 8.4. Inference in Graphical Models

### 8.4.1 Inference on a chain

我们这里只需要讨论无向图。因为有向图可以直接转化为无向图。We have
already seen that the directed chain can be transformed into an equivalent undirected chain. Because the directed graph does not have any nodes with more than one parent, this does not require the addition of any extra links, and the directed and undirected versions of this graph express exactly the same set of conditional independence statements.

![](Pasted%20image%2020210426121756.png)

上边的chain可以写成下面的potential，同时我们假设变量都是离散的，每个变量有K个状态
![](Pasted%20image%2020210426121959.png)

Let us consider the inference problem of **finding the marginal distribution $p(x_n)$** for a specific node $x_n$ that is part way along the chain.

为了避免指数级的求和运算， exploiting the conditional independence properties of the graphical model

思想：
把![](Pasted%20image%2020210426122616.png)代入![](Pasted%20image%2020210426122630.png)，然后观察

先看the summation over$x_N$, 我们知道$x_N$只与$x_{N-1}$相关，因此我们可以通过$x_N$与$x_{N-1}$的potential得到二者的joint，再在$x_N$上求和，就能得到$x_{N-1}$的分布
![](Pasted%20image%2020210426122948.png)

以此类推，Because each summation effectively removes a variable from the distribution, this can be viewed as the removal of a node from the graph.

we can express the desired marginal in the form
![](Pasted%20image%2020210426123338.png)
这个式子没有增加什么新东西，只是把(8.50)重新排序
这个式子的复杂度为$O(NK^2)$


#### passing of local messages
We now give a powerful interpretation of this calculation in terms of the **passing of local messages** around on the graph.

marginal $x_N$ 可以分解为两项之积以及normalization term
![](Pasted%20image%2020210426153533.png)

We shall interpret $µ_α(x_n)$ as a message passed forwards along the chain from node $x_{n−1}$ to node $x_n$. 
Similarly, $µ_β(x_n)$ can be viewed as a message passed backwards along the chain to node $x_n$ from node $x_{n+1}$. 
注意$\mu$只代表相邻一项传过来的信息
Note that each of the messages comprises a set of K values, one for each choice of xn, and so the product of two messages should be interpreted as the **point-wise multiplication** of the elements of the two messages to give another set of K values.

因此可以得出递归式：
![](Pasted%20image%2020210426154820.png)
![](Pasted%20image%2020210426154829.png)
因此因为头和尾没有μ，因此可以递归计算，如：
![](Pasted%20image%2020210426160015.png)
![](Pasted%20image%2020210426155804.png)
注意脚标：The outgoing message $µ_α(x_n)$ in (8.55) is obtained by multiplying the incoming message $µ_α(x_{n-1})$ by the local potential involving **the node variable and the outgoing variable** and then summing over **the node variable**.

这个图叫做**Markov chains**，and the corresponding message passing equations represent an example of the **ChapmanKolmogorov equations** for Markov processes

如果想得到chain上每个变量的marginal，就可以先从$x_N$开始求出所有的$µ_\beta(x_i)$，从从$x_1$开始求出所有的$µ_\alpha(x_i)$。把所有的结果存起来，然后再算marginal。

***
#### Note: observation
如果有某些变量被观测：If some of the nodes in the graph are observed, then **the corresponding variables are simply clamped to their observed values and there is no summation**.
注意这个指示函数$I$是和summation同时出现的，举个例子：我们要算$\sum_{x_1}\sum_{x2}f(x_1,x_2)$，然后我们观测了$x_2$为$\hat{x_2}$
直接代入的话为$\sum_{x_1}f(x_1,\hat{x_2})$,
如果写成summation+I的话，就是$\sum_{x_1}\sum_{x2}f(x_1,x_2)\cdot I(x_2,\hat{x_2})$,
此时把summation$\sum_{x2}$展开，此时只有$x_2=\hat{x_2}$的项目会乘1，其他项会乘0.然后再把每一项相加，最后得到的结果就是$\sum_{x_1}f(x_1,\hat{x_2})$，和直接代入的结果相同

![](Pasted%20image%2020210430165125.png)
***
相邻两个变量的joint distribution：$p(x_{n-1}, x_n)$
This is similar to the evaluation of the marginal for a single node, except that there are now two variables that are not summed out.
![](Pasted%20image%2020210426163108.png)
![](Pasted%20image%2020210426163122.png)


### 8.4.2 Trees

* In the case of an undirected graph, a tree is defined as:
**a graph in which there is one, and only one, path between any pair of nodes.**
Such graphs therefore do not have loops. 

* In the case of directed graphs, a tree is defined such that:
**there is a single node, called the root, which has no parents, and all other nodes have one parent**

![](Pasted%20image%2020210426164355.png)

the moralization step will not add any links as all nodes have at most one parent, and as a consequence the corresponding moralized graph will be an undirected tree

**If there are nodes in a directed graph that have more than one parent, but there is still only one path (ignoring the direction of the arrows) between any two nodes**, 
then the graph is a called a **polytree**,

Such a graph will have more than one node with the property of having no parents, and furthermore, the corresponding moralized undirected graph will have loops.

### 8.4.3 Factor graphs
sum-product algorithm 适用于undirected and directed trees and to polytrees
我们也可以把它们统一成一种结构：factor graph

有向图和无向图都可以把所有变量的总joint分解为function of subset的乘积。
Factor graphs make this decomposition explicit **by introducing additional nodes for the factors themselves** in addition to the nodes representing the variables.

![](Pasted%20image%2020210426170222.png)

这部分我们用$\mathrm{x_s}$表示变量的子集，用$x_i$表示单一变量（和以前不同，以前$x_i$也可以表示变量的集合）

对于有向图，$f_s(\mathrm{x_s})$ are local conditional distributions. 
对于无向图，$f_s(\mathrm{x_s})$ are potential functions over the maximal cliques (the normalizing coefficient 1/Z can be viewed as a factor defined over the empty set of variables)


具体画法：
* there is a node (depicted as usual by a circle) for every variable in the distribution, as was the case for directed and undirected graphs. 
* There are also additional nodes (depicted by small squares) for each factor fs(xs) in the joint distribution. 
* Finally, there are undirected links connecting each factor node to all of the variables nodes on which that factor depends.

比如下面的分布可以画成这样：
![](Pasted%20image%2020210426190641.png)
![](Pasted%20image%2020210426190656.png)

在上面这个例子中，$f_1$和$f_2$其实可以合并成potential，$f_3$和$f_4$也可以。但是在factor graph中keeps such factors explicit and so is able to convey more detailed information about the underlying factorization.

Factor graphs are said to be **bipartite**(二分图) because they consist of two distinct kinds of nodes, and **all links go between nodes of opposite type. **

对于无向图，我们对每个maximum clique建立一个factor node，并把f就设为clique potentials。同一个图可能有不同的factor graph：
![](Pasted%20image%2020210426193410.png)

对于有向图，create factor nodes corresponding to the conditional distributions, and then finally add the appropriate links.同一个图可能有不同的factor graph
![](Pasted%20image%2020210426194452.png)

如果我们对directed tree或undirected tree 做moralize，结果仍然是tree(in other words, the factor graph will have no loops, and there will be one and only one path connecting any two nodes)

需要注意的是，对于**directed polytree**，我们在转化为undirected graph时，是有loop的。但是在moralization中, conversion to a factor graph **again results in a tree**. 
![](Pasted%20image%2020210426204607.png)
In fact, local cycles in a directed graph due to links connecting parents of a node can be removed on conversion to a factor graph by defining the appropriate factor function. 此外，有向图中"head to head"产生的local cycles，如果选择合适的factor function，就能在factor graph中去掉：
![](Pasted%20image%2020210426204707.png)


### 8.4.4 The sum-product algorithm
**"evaluating local marginals over nodes or subsets of nodes"**

在本章中，我们假设变量都是离散的，这样我们就可以用sum来做marginalize。但事实上sum-product algorithm对于linear-Gaussian models也适用

在directed graph中还有一个用于exact inference的算法，belief propagation，可以看作sum-product algorithm的一个special case。
Here we shall consider only the sum-product algorithm because it is simpler to derive and to apply, as well as being more general.

我们假设原本的graph是an undirected tree or a directed tree or
polytree，**so that the corresponding factor graph has a tree structure**

Our goal：
1. to obtain an efficient, exact inference algorithm for finding marginals
2. in situations where several marginals are required to allow computations to be shared efficiently

#### finding the marginal p(x) for particular variable node x

目前，我们假设所有变量都是hidden的。

By definition, the marginal is obtained by summing the joint distribution over all variables except x so that
![](Pasted%20image%2020210426211321.png)

Idea:
to substitute for p(x) using the factor graph expression (8.59) 
![](Pasted%20image%2020210426211422.png)
and then interchange summations and products in order to obtain an efficient algorithm

回想到bipartite的性质，与x相连的一定是f节点
再回想到tree的性质，任意两个节点之间只有一个path

因此把与x相连的节点从x与f的连接处断开，分成若干个group，各个group并不相连
![](Pasted%20image%2020210426223719.png)

回想factor graph的定义，joint可以写成product of functions of subset
![](Pasted%20image%2020210426224015.png)

![](Pasted%20image%2020210426224126.png)

ne(x) denotes the set of factor nodes that are neighbours of x, and Xs denotes the set of all variables in the subtree connected to the variable node x via the factor node fs, and Fs(x,Xs) represents the product of all the factors in the group associated with factor fs.

各个group并不相连，因此可以写成各个group的乘积

代入到p(x)的表达式中，利用乘法分配律交换乘法与加法
![](Pasted%20image%2020210426225256.png)
functions of each group 的乘积，对于所有其他变量求sum
->
对于每个function of each group，对于所有其他变量求sum，然后再乘起来
(而在因为group彼此不相连，在$group_i$上，对于所有变量sum等于对$group_i$内部的变量sum，也就是$X_s$，所以上图中的summation可以写成$X_s$)

将group内部的summation定义为**messages** from the factor nodes $f_s$ to the variable node x
![](Pasted%20image%2020210426230211.png)
We see that the required marginal p(x) is given by the product of all the incoming messages arriving at node x.
marginal p(x)就是所有传到node x的message的乘积

$F_s(x,X_s)$ is described by a factor (sub-)graph and so **can itself be factorized**. In particular, we can write
![](Pasted%20image%2020210426231032.png)
where, for convenience, we have denoted the variables associated with factor $f_s$,in addition to x,by $x_1,...,x_M$，同时也可以写成$\mathrm{x}_s$
（因为$F_s(x,X_s)$也是一个factor (sub-)graph，因此也可以写成product of functions of subset的形式，上面的式子只是一种比较有用的形式）

![](Pasted%20image%2020210426230951.png)

![](Pasted%20image%2020210426232008.png)
注意这里的$f_s(...)$就是各个f节点所对应的函数
观察到上面的这个
![](Pasted%20image%2020210426232429.png)
又是一个summation on group，因此还是可以写成message的形式，只不过这次由(图8.47)可以看出，message是由x node流向f node的。
![](Pasted%20image%2020210426232909.png)
还是由于tree的性质，(图8.47)中的每个group仍然是互不相连的

然后再对$G_m(x_m,X_{sm})$分解，
![](Pasted%20image%2020210426233641.png)
where the product is taken over all neighbours of node xm except for node $f_s$
![](Pasted%20image%2020210426233519.png)

同时我们观察到，$F_l(x_m,X_{ml})$和最初求p(x)的因子是一样的
Note that each of the factors $F_l(x_m,X_{ml})$ represents a subtree of the original graph of precisely the same kind as introduced in (8.62).
![](Pasted%20image%2020210426233959.png)


Substituting (8.68) into (8.67), we then obtain
![](Pasted%20image%2020210426234210.png)
where we have used the definition (8.64) of the messages passed from factor nodes to variable nodes.

**Thus to evaluate the message sent by a variable node to an adjacent factor node along the connecting link, we simply take the product of the incoming messages along all of the other links.**

Note:
* any variable node that has only two neighbours performs no computation but simply passes messages through unchanged
* a variable node can send a message to a factor node once it has received incoming messages from all other neighbouring factor nodes


Each of these messages can be computed recursively in terms of other messages. In order to start this recursion, we can view the node x as the root of the tree and begin at the leaf nodes.

现在就来考虑leaf nodes的计算，
* if a leaf node is a variable node, then the message that it sends along its one and only link is given by
	![](Pasted%20image%2020210427000131.png)
	因为此时F代表在一个empty的group上面summation，结果为1
	![](Pasted%20image%2020210427000218.png)
* if the leaf node is a factor node, we see from (8.66) that the message sent should take the form
	![](Pasted%20image%2020210427000320.png)
	因为此时G也是在一个empty的group上面summation，结果为1，而前边对各种x进行summation之后，就变成了f(x)
	![](Pasted%20image%2020210427000334.png)
	
	
![](Pasted%20image%2020210427000511.png)

总结：
1. viewing the variable node x as the root of the factor graph and initiating messages at the leaves of the graph using (8.70) and (8.71)
	![](Pasted%20image%2020210427105642.png)
	![](Pasted%20image%2020210427105652.png)
1. The message passing steps (8.66) and (8.69) are then applied recursively until messages have been propagated along every link, and the root node has received messages from all of its neighbours
	![](Pasted%20image%2020210427105714.png)
	![](Pasted%20image%2020210427105804.png)
1. Once the root node has received messages from all of its neighbours, the required marginal can be evaluated using (8.63).
	![](Pasted%20image%2020210427105830.png)


#### find the marginals for every variable node

We can obtain a much more efficient procedure by ‘overlaying’ these multiple message passing algorithms to obtain the general sum-product algorithm as follows

1. 任意选定一个node作为root
2. 从leaf把message传到root
3. root可以从所有的邻居那里收到message，也就能把信息再发给每个邻居

By now, a message will have passed in both directions across every link in the graph

#### find the marginal distributions $p(\mathrm{x_s})$ associated with the sets of variables belonging to each of the factors

it is easy to see that the marginal associated with a factor is given by the **product of messages arriving at the factor node and the local factor at that node**
![](Pasted%20image%2020210427111735.png)
in complete analogy with the marginals at the variable nodes

回想(8.66)，这两者都是要求$f_s$节点所代表的"message".只不过(8.66)严格意义上的message需要对$\mathrm{x_s}$summation，而我们想求$p(\mathrm{x_s})$，当然就不用求和了
![](Pasted%20image%2020210427112019.png)

If the factors are parameterized functions and we wish to learn the values of the parameters using the EM algorithm, then these marginals are precisely the quantities we will need to calculate in the E step, as we shall see in detail when we discuss the hidden Markov model in Chapter 13.

#### a different view
因为从variable node指向factor node，就只是单纯的相乘
因此可以忽视掉所有的variable node，将max-product只看作是一个factor node之间传递message的过程
The sum-product algorithm can be viewed purely in terms of messages sent out by factor nodes to other factor nodes.

![](Pasted%20image%2020210427113248.png)

#### normalization
如果factor graph由有向图转换而来，the joint distribution is already correctly normalized
如果factor graph由无向图转换而来，in general there will be an unknown normalization coefficient 1/Z. 

As with the simple chain example of Figure 8.38, this is easily handled by working with an unnormalized version $\widetilde{p}(x)$ of the joint distribution,
where $p(x) = \widetilde{p}(x)/Z$
![](Pasted%20image%2020210427113900.png)
We first run the sum-product algorithm to find the corresponding **unnormalized marginals $p(x_i)$**. The coefficient $1/Z$ is then easily obtained by normalizing any one of these marginals, and this is computationally efficient because the normalization is done over a single variable rather than over the entire set of variables as would be required to normalize $p(x)$ directly.
也就是，对于无向图，在marginalize之前的所有计算中，joint都可以不用normalize。等到最终求出margin之后，对于这一个变量求和得出Z就可以了

#### example
![](Pasted%20image%2020210427130347.png)
这里这个graph并不一定是从有向图还是无向图得来，因此不一定normalize
这个graph的**unnormalized** joint distribution为：
![](Pasted%20image%2020210427130700.png)

![](Pasted%20image%2020210427131001.png)
![](Pasted%20image%2020210427131217.png)
![](Pasted%20image%2020210427131229.png)

此时，每个link上面两个方向的message就都算出来了
最后，根据
![](Pasted%20image%2020210427131539.png)
求出$x_2$的unnormalized margin：
![](Pasted%20image%2020210427131609.png)
在此之上对$x_2$求和，就能求出normalization term Z

#### observed variable


In most practical applications, a subset of the variables will be observed, and we wish to calculate posterior distributions conditioned on these observations.

这段要类比于8.4.1中对于chain的observe看，
![](Pasted%20image%2020210427202425.png)
注意此时得出的结果是unnormalized的


### 8.4.5 The max-sum algorithm
sum-product algorithm用于take a joint distribution $p(\mathrm{x})$ expressed as a factor graph and efficiently find marginals over the component variables
一般我们还有两个常见的任务：
* find a setting of the variables that has the largest probability
* find the value of that probability

这两个工作可以用**max-sum**, which can be viewed as an application of **dynamic programming** in the context of graphical models

最简单的思路就是，对每个变量x都跑一遍max-product，然后在各自的marginal上边找到最大值 $x_i^*$. 但是this would give the set of values that are **individually** the most probable.

In practice, we typically wish to find the **set of values** that **jointly** have the largest probability, in other words the vector $x_{max}$ that maximizes the joint distribution, so that
![](Pasted%20image%2020210427205442.png)

也就是我们要找**一组值**，让joint $p(\mathrm{x})$达到 max，而不是在每一维度找一个max然后拼起来


#### find the maximum of the joint distribution (by propagating messages from the leaves to an arbitrarily chosen root node)
假设一共有M个变量，可以把max写成展开的形式：
![](Pasted%20image%2020210427205941.png)

and then substitute for $p(\mathrm{x})$ using its expansion in terms of a product of factors。然后可以对$p(\mathrm{x})$进行分解

在max-product中，我们利用乘法分配律交换了乘法与求和
而在这里max-sum中，我们利用max的性质交换乘法与max
![](Pasted%20image%2020210427210433.png)

先考虑chain上的情况，因为是factor graph，可以拆分
![](Pasted%20image%2020210427211548.png)
![](Pasted%20image%2020210427211852.png)

我们可以看到交换之后的式子is easily interpreted in terms of messages passed from node xN backwards along the chain to node x1. （在引入factor graph之前的那部分讨论过，chain的message就是potential）

下面从chain推广到tree，把
![](Pasted%20image%2020210427223345.png)
代入到max的性质中
The structure of this calculation is identical to that of the sum-product algorithm, and so we can simply translate those results into the present context.

In particular, suppose that we designate a particular variable node as the ‘root’ of the graph. 
Then we start a set of messages propagating inwards from the leaves of the tree towards the root, with each node sending its message towards the root once it has received all incoming messages from its other neighbours. 
The final maximization is performed over the product of all messages arriving at the root node, and gives the **maximum** value for $p(\mathrm{x})$
![](Pasted%20image%2020210427224107.png)
观察sum-product得到的式子，这个式子是由乘法分配律得到的。max-product的推导在下面。
This could be called the max-product algorithm and is identical to the sum-product algorithm except that **summations are replaced by maximizations**.

***
Q：那这样岂不是应该把上面的式子的求和换成max么，不应该是max-product么，为什么是max-sum？
通常对p(x)求ln，防止underflow。In practice, products of many small probabilities can lead to numerical underflow problems, and so it is convenient to work with the logarithm of the joint distribution.
因为lnx单调，所以max和ln可以交换
![](Pasted%20image%2020210427225220.png)
此时max的性质还在，只不过从乘法变成加法
![](Pasted%20image%2020210427225331.png)

**Thus taking the logarithm simply has the effect of replacing the products in the max-product algorithm with sums, and so we obtain the max-sum algorithm**

类比sum-product的结论，我们可以直接写出max-sum的式子：
![](Pasted%20image%2020210427231754.png)
![](Pasted%20image%2020210427231839.png)
while at the root node the maximum probability can then be computed, by analogy with (8.63), using
![](Pasted%20image%2020210427232538.png)

由此就得到了joint的max

#### finding the configuration of the variables for which the joint distribution attains this maximum value
我们想要得到与上面的$p^{max}$对应的$x^{max}$
![](Pasted%20image%2020210427233031.png)
At this point, we might be tempted simply to continue with the message passing algorithm and send messages from the root back out to the leaves, using (8.93) and (8.94), then apply (8.98) to all of the remaining variable nodes. However, because we are now maximizing rather than summing, it is possible that there may be multiple configurations of x all of which give rise to the maximum value for p(x).In such cases, this strategy can fail because **it is possible for the individual variable values obtained by maximizing the product of messages at each node to belong to different maximizing configurations, giving an overall configuration that no longer corresponds to a maximum**.

The problem can be resolved by adopting a rather different kind of message passing from the root node to the leaves

let us return once again to the simple chain example of N variables x1,...,xN each having K states
Suppose we take node xN to be the root node. Then in the first phase, we propagate messages from the leaf node x1 to the root node using
![](Pasted%20image%2020210427234256.png)
![](Pasted%20image%2020210427234305.png)
然后就能得到The most probable value for xN
![](Pasted%20image%2020210427234338.png)

现在我们想得到the states of the previous variables that **correspond to the same maximizing configuration**.

This can be done by **keeping track of which values of the variables gave rise to the maximum state of each variable**, in other words by storing quantities given by
![](Pasted%20image%2020210427234628.png)


下面来解释一下这个式子：
![](Pasted%20image%2020210428000602.png)
Note that this is not a probabilistic graphical model because the nodes represent individual states of variables
For each state of a given variable, there is a unique state of the previous variable that maximizes the probability 
Once we know the most probable value of the final node xN, we can then simply follow the link back to find the most probable state of node xN−1 and so on back to the initial node x1

This corresponds to propagating a message back down the chain using
![](Pasted%20image%2020210428001101.png)
and is known as **back-tracking**.


如果我们正向max-sum之后，反向传播，然后对每个变量套用这个式子：
![](Pasted%20image%2020210428002742.png)
最后选出的组合可能分布在(图8.53)的几条不同的路径中，最终就导致不是global maximum

extension to a general tree-structured factor graph：
换句话说，在正向message传播的时候，我们需要算这个东西，
![](Pasted%20image%2020210428011457.png)
在对M个变量逐个max的时候，我们记下每个变量满足这个式子的值
![](Pasted%20image%2020210428011627.png)
然后我们在正向传播结束之后通过这个式子得到了$p^{max}$
![](Pasted%20image%2020210428012205.png)
此时我们存下来的那些值就自动对应$p^{max}_1,...,p^{max}_M$


An important application of this technique is for finding the most probable sequence of hidden states in a hidden Markov model, in which case it is known as the Viterbi algorithm

#### observation
The observed variables are clamped to their observed values, and the maximization is performed over the remaining hidden variables. This can be shown formally by including identity functions for the observed variables into the factor functions, as we did for the sum-product algorithm.

### 8.4.6 Exact inference in general graphs
sum-product和max-sum适用于tree，然而实践中we have to deal with graphs having loops
将tree上面的inference推广到任意拓扑结构的算法称为junction tree algorithm
大概步骤：
1. 把有向图转化为无向图，无向图不用变
2. Next the graph is *triangulated*, which involves finding chord-less cycles containing four or more nodes and adding extra links to eliminate such chord-less cycles
	![](Pasted%20image%2020210428133838.png)
	在上面这个图中，A-D-B-C-A为chordless cycle，此时要在AB或CD任意加一条边
	注意potential还是原来的函数，只不过是按新的结构进行分解
3. 用这个triangulated undirected graph建立a new tree-structured undirected graph called a join tree
	join tree中的每个节点代表无向图中的一个maximal clique，有公共变量的maximal clique之间有link
	* 这个连接各个clique形成树的过程，要满足一定的条件，使得最后形成的树是maximal spanning tree
	* 其中link的weight为两个clique共有变量的个数，tree的weight是树上所有link的weight之和
	* If the tree is condensed, so that any clique that is a subset of another clique is absorbed into the larger clique, this gives a **junction tree**. 
	* **As a consequence of the triangulation step**, the resulting tree satisfies the **running intersection property**, which means that if a variable is contained in two cliques, then it must also be contained in every clique on the path that connects them.
4. a two-stage message passing algorithm, essentially equivalent to the sum-product algorithm, can now be applied to this junction tree in order to find marginals and conditionals

虽然前面很复杂，但这个算法的核心仍然是交换乘法和加法，使得我们可以先summation出局部的message，再把message传递相乘。而不是直接对joint 进行summation。
at its heart is the simple idea that we have used already of exploiting the factorization properties of the distribution to **allow sums and products to be interchanged so that partial summations can be performed, thereby avoiding having to work directly with the joint distribution**

局限性：
Unfortunately, the algorithm must work with the joint distributions **within each node** (each of which corresponds to a clique of the triangulated graph) and so the computational cost of the algorithm is determined by the number of variables in the largest clique，并且会指数增长


### 8.4.7 Loopy belief propagation

在approximate inference中，需要用到chapter10中的variational methods和chapter11的Monte Carlo methods。
这里先简单介绍一种用于graph with loops 的approximate方法
思想：
The idea is simply to apply the sum-product algorithm even though there is no guarantee that it will yield good results.
因为graph现在有cycles了，information can flow many times around the graph. For some models, the algorithm will converge, whereas for others it will not.
(Mark，这里跳了一段)
### 8.4.8 Learning the graph structure
之前我们都是假设graph是已知的，
理论上我们可以为graph的结构设一个prior，然后利用posterior来进行预测：
![](Pasted%20image%2020210428144111.png)
也就是make predictions by averaging with respect to this distribution
然而这样计算量太大，一般都会采用heuristics来筛选graph的结构

