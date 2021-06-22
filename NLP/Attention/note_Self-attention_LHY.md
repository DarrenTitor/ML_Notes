# # Self-attention
[video: 【機器學習2021】自注意力機制 (Self-attention) (上)](https://www.youtube.com/watch?v=hYdO9CscNes)
## 场景
### input
input是vector set，而且set大小不固定

注意这里提到graph可以看作一个vector set
![](Pasted%20image%2020210622213854.png)

### output
* 每个vector有一个label: 词性分析 
* 整个sequence有一个label：情感分析
* 变长label-seq2seq：翻译


## Self-attention
下面讨论Sequence Labeling的场景。属于上面的第一种，每个vector有一个label

![](Pasted%20image%2020210622214513.png)
MLP不行，因此要考虑context
![](Pasted%20image%2020210622214553.png)
可以用MLP+window，但是缺点是不能考虑到整个sequence

因此要用到self-attention
![](Pasted%20image%2020210622214931.png)
self-attention会输出与input一样长的output，而这个过程是考虑了整个sequence的。而它输出的vector的个数与它的输入相同。

self-attention可以叠加多次
![](Pasted%20image%2020210622215241.png)


### How
以生成$b_1$这个值为例：
1. 找到与$a_1$相关的向量，每个word与$a_1$的相似度用$\alpha$表示。

	![](Pasted%20image%2020210622220200.png) 
	计算$\alpha$的方法有很多种：
	![](Pasted%20image%2020210622220301.png)
	比如dot product就是把两个word分别乘上两个矩阵，所得结果作inner product得到$\alpha$。transformer中用的就是这种
2.  $a_1$与$W^q$相乘得到一个'query'$q^1$，其他的$a^j$都分别与$W^k$相乘得到'key'$k^j$。然后分别作inner product得到attention score$\alpha_{1,j}$.
	![](Pasted%20image%2020210622220928.png)
		实际操作中，自己跟自己也要做attention
	![](Pasted%20image%2020210622221156.png)
3. 通过softmax(不一定非要softmax，可以尝试其他的)
	![](Pasted%20image%2020210622221301.png)
4. 把每个**word**都乘上$W^v$得到$v^i$，然后把$v^i$对于attention score$\alpha_{1,j}$加权求和，得到$b^1$。注意所有的b的生成没有先后顺序，它们是同时生成的。

	![](Pasted%20image%2020210622221452.png)
<br>



### How (matrix)
从矩阵的角度，把输入的vector set拼成一个矩阵，分别乘上三个不同的W就得到了Q,K,V.
![](Pasted%20image%2020210622230437.png)

接下来，每一个q会和每一个k做inner product，得到attention score。
每一组的操作可以看成是一个矩阵和一个向量相乘
![](Pasted%20image%2020210622230903.png)
总体上，就是两个矩阵相乘。然后如果选用softmax，就对结果进行column-wise的softmax。
![](Pasted%20image%2020210622231025.png)

接下来进行weighted sum，
![](Pasted%20image%2020210622231332.png)
矩阵O的每一个column就是self-attention的输出。

总结：
![](Pasted%20image%2020210622231525.png)
需要learn的只有$W^q,W^k,W^v$.


### Multi-head Self-attention

假设我们对于'相关'，有多种定义。(这里假设有两种，也就是head个数为2)
因此我们生成两组q,v,k，分别算出它们的attention score
![](Pasted%20image%2020210622232506.png)
之后把这两个attention score拼接起来，乘上一个矩阵，得到最终的attention score。也就是learn一个transformation。
![](Pasted%20image%2020210622232611.png)


### Positional Encoding
在前面的self-attrntion中，没有position information。

思路很简单，在每个input vector上面加一个unique positional vector $e^i$
这个e最早是hand-crafted的，现在是通过设置特定的函数产生，甚至可以learn。
![](Pasted%20image%2020210622233224.png)
![](Pasted%20image%2020210622233305.png)

### Application
#### NLP
* Transformer
* BERT

#### Speech
self-attention的复杂度是sequence长度的平方，因此在语音任务中常常要根据对人物的理解减小attention的范围
![](Pasted%20image%2020210622233816.png)

#### Image
![](Pasted%20image%2020210622233936.png)
![](Pasted%20image%2020210622233956.png)

#### Graph
之前提到，node可以作为input的vector。
那么edge的信息可以在计算attention时反映出来。
在计算attention时，我们可以利用现有的graph，只用相连的node计算attention score
![](Pasted%20image%2020210623000050.png)
### self-attention v.s. CNN
CNN的receptive field只有一个filter大小，
self-attention则关注整张图片，而且是learnable的
![](Pasted%20image%2020210622234146.png)

而我们知道越是flexable的model越需要更多的data，才能达到比较好的效果。
因此在数据量少的时候，CNN比self-attention的表现更好。

### self-attention v.s. RNN
 
 首先我们可能观察到，RNN只能考虑到前半段sequence，self-attention能考虑到整个sequence。
 但如果我们用双向RNN的话，也是可以考虑整个sequence的。
 ![](Pasted%20image%2020210622235002.png)
 
 更主要的区别其实是：
 1. RNN很难考虑到距离很远的word，必须要把那个word的信息一直存在memory中不忘记才行。而self-attention没有这个问题，只要query和key能match，就能考虑到。
 2. RNN比如其次计算，nonparallel。self-attention所有的output同时生成，可以parallel。
 ![](Pasted%20image%2020210622235252.png)
 
 
### More
self-attention的计算量很大，因此产生了很多变体减少计算量，当然性能也有一定下降。
![](Pasted%20image%2020210623000309.png)
 
 
 
 
 
