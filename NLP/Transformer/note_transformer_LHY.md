# Transformer

## Seq2seq
output的长度由model自己来决定。

NLP的许多问题都可以看作是一个QA系统，而QA系统就可以用一个seq2seq的模型来实现。
![](Pasted%20image%2020210624014617.png)
但往往针对问题采用特定的模型表现会更好。


比如在文法解析的任务中，我们想要输出的不是序列，是一个文法的树，其实也可以把树硬写成sequence，然后用seq2seq。

seq2seq还可以用于multi-label classification
![](Pasted%20image%2020210624015159.png)

甚至可以做目标识别：
![](Pasted%20image%2020210624015327.png)



## Encoder
input一排向量，output一排同樣長度的向量
transformer中的encoder用的就是self-attention

![](Pasted%20image%2020210627163724.png)
上边图中的每一个block不是一个layer，而是很多个layer。
每一个layer都是先经过一个self-attention，然后再把输出经过fully-connected layer处理之后得到block的输出。


在transformer的原始设计中，block中的self-attention要用到residual连接，连接之后加和的结果要经过一个normalization。
这个normalization不是batch normalization，而是更简单的layer normalization。
layer normalization不用考虑batch，只考虑向量本身。输入一个向量，
这里与batch normalization作区分。batch normalization是对不同的example的同一个dimention计算mean和deviation。
layer normalization是对同一个example不同的dimention计算mean和deviation。然后进行normalize。
![](Pasted%20image%2020210627165328.png)



layer normalization之后的输出作为FC的输入，这里还要进行一次residual操作。然后再layer normalization一次得到一个block的输出。
![](Pasted%20image%2020210627165419.png)


![](Pasted%20image%2020210627165854.png)
然后可以和原论文中的图对照一下，注意这里其实还有一个positional encoding，因为self-attention没有位置的资讯，要通过positional encoding来引入。
同时注意这里的self-attention是multi-head的。

BERT其实就是transformer的encoder，结构是一样的。


上面说的只是原始论文的设计，还可以有其他的设计。
比如我们可以改变layer normalization在网络中的位置。
（下图中的(b)）
![](Pasted%20image%2020210627170036.png)
图中的第二篇论文解释了为什么batch normalization不如layer normalization。

## Decoder
### Decoder (Autoregressive)

在decoder接收到的输入中，要有一个special token BEGIN用来标识一个sequence的开头。

![](Pasted%20image%2020210627190948.png)

BEGIN通过decoder会得到一个输出，这个输出的维度很大，维数等于vocabulary的大小。
通常输出之前会经过一个softmax，因此输出是一个distribution，和为1.
然后选择概率值最大的一维"机"作为输出。

接下来，decoder用BEGIN和"机"这两个向量作为输入，得到输出"器"
![](Pasted%20image%2020210627191200.png)

decoder会把自己之前的输出当作接下来的输入。（同时也会接收到来自encoder的输出）
![](Pasted%20image%2020210627191427.png)

可见如果其中有某个输出是错误的，decoder就会把错误的输出当作输入。

会不会产生error propagation呢？
有可能，稍后讲解决方法。


#### 内部结构
![](Pasted%20image%2020210627191845.png)


![](Pasted%20image%2020210627191944.png)
对比一下encoder和decoder的结构，可以看到如果把decoder的其中一部分遮住，二者的结构是很像的。
![](Pasted%20image%2020210627192218.png)
但是注意到decoder的multi-head self-attention上面多了一个"masked".

下面就来讲这个masked是什么：
这个是原本的self-attention，每一个输出b都看过了整个序列
![](Pasted%20image%2020210627193101.png)

在masked self-attention中，每一个输出都不能看到它右边的输入，比如$b_2$只能看到$a_1$和$a_2$。
![](Pasted%20image%2020210627193337.png)



换句话说，在普通的self-attention中，$a_2$产生的query会和所有的key作inner product，得到softmax得到attention score，然后再用v和attention score作weighted sum得到$b_2$。
![](Pasted%20image%2020210627193437.png)

而在masked self-attention中，$q_2$只和$k_1$和$k_2$作inner product，最终计算出$b_2$。
![](Pasted%20image%2020210627194015.png)
**(这里虽然mask了，但也还是输出长度等于输入长度)**。
Why masked?
在decoder中，$a_1,a_2,a_3,a_4$是先后产生的。当想要计算$b_2$的时候，实际上根本就还没产生$a_3,a_4$。因为decoder的输入是一个一个产生的，因此它每次只能考虑它左边的东西。
不同于encoder，在encoder中，$a_1,a_2,a_3,a_4$是我们本身就有的文本，当然可以一次性放进input。

***
这里还有一个重要的问题，decoder必须决定输出的sequence的长度，需要知道自己什么时候停下来。
![](Pasted%20image%2020210627195655.png)
因此要准备一个特殊的符号END
![](Pasted%20image%2020210627195734.png)
![](Pasted%20image%2020210627195757.png)

### Decoder (Non-Autoregressive)(NAT)
![](Pasted%20image%2020210627200539.png)




## Encoder-Decoder
连接encoder和decoder的桥梁，就是之前遮住的那个部分，叫做cross-attention，可以看到encoder提供了2个输入，decoder提供了1个输入.
![](Pasted%20image%2020210627200809.png)

![](Pasted%20image%2020210627201328.png)
decoder中masked self-attention的输出乘以一个矩阵之后得到query，然后分别和encoder的输出变换得到的k计算attention score，然后score和encoder的输出变换得到的v作weighted sum(可能有weight可能经过softmax)，得到最终的v，然后再把这个v送入后面的fully connected network。这就是cross attention的工作。
因此encoder提供k和v，decoder的前半部分提供q，最终得到的输出再送入fc层。
![](Pasted%20image%2020210627202120.png)



在之前的例子，也就是原始的transformer中，decoder的许多层corss-attention都是接收encoder的最后一层的输出。
但其实后来出现了各种各样的连接方法
![](Pasted%20image%2020210627202802.png)

## Training

每次decoder在产生一个word的时候，其实就是做了一个分类。
![](Pasted%20image%2020210627203157.png)
我们希望所有分类问题的cross-entropy的总和最小，注意不光要计算所有的word，还要计算END的cross-entropy。
![](Pasted%20image%2020210627203329.png)

Note:
**和之前讲的inference阶段的做法不同，在training阶段，decoder的输入是ground truth.**
(在inference阶段，decoder看到的是自己的output)
![](Pasted%20image%2020210627203800.png)
但是这样一来，training阶段看到ground truth，inference阶段看到自己的输出，这之间有一个mismatch。(之后讲解决方案)


## Tips
### Copy Mechanism
在刚才的讨论中，我们都要求decoder自己创造输出。
但是在有些任务中，我们只需要model复制input sequence中的一部分。
![](Pasted%20image%2020210627204410.png)
![](Pasted%20image%2020210627204437.png)
![](Pasted%20image%2020210627204609.png)

### Guided Attention
![](Pasted%20image%2020210627211251.png)
强迫attention计算的过程，比如从左向右。

### Beam Search
![](Pasted%20image%2020210627211712.png)
(有时候有用，有时候没用)
在task的正确答案很明确的时候，可能beam search比较好，比如语音辨识。
当task的正误不明确的时候，不用beam search比较好，比如文本生成。

在generating sequence的任务中，有时要在decoder中加入randomness，比如在decoder的input中加入noise。
![](Pasted%20image%2020210627211908.png)

TTS(语音合成)在测试的时候也要加入一些noise，结果才会好。



### Optimizing Evaluation Metrics
![](Pasted%20image%2020210627212950.png)
在训练的时候，我们是minimize每个词的cross entropy，
但是在测试的时候我们是算预测得到的句子和真实的句子之间的BLEU score。

那么minimize cross entropy真的可以maximize BLEU score吗？
不一定。

因此应该在validation的时候使用BLEU score。

可以在training的时候用BLEU score吗？
不能，因为BLEU score不可微。
![](Pasted%20image%2020210627213417.png)



### Scheduled Sampling
接下来讨论之前提到的训练和测试mismatch的问题：exposure bias
![](Pasted%20image%2020210627213538.png)
解决的思路：
给decoder看一些错误的东西
![](Pasted%20image%2020210627213659.png)
原始的Scheduled Sampling会影响到transformer的并行化，因此后来提出了专门用于transformer的Scheduled Sampling。

