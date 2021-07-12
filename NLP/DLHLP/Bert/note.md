# BERT

## What is pre-train model

最早的pre-train:
Represent each token by a embedding vector
比如word2vec、glove。
缺点：不知道上下文

变种：
输入字母，输出word的embedding：
![](Pasted%20image%2020210704005152.png)
中文：
![](Pasted%20image%2020210704005311.png)

Contextralized word embedding:
看过整个句子再给embedding
![](Pasted%20image%2020210704005611.png)
![](Pasted%20image%2020210704005819.png)
从上图可见BERT可以辨别多义词

小一点的BERT：
![](Pasted%20image%2020210704010104.png)
这些小模型用到的压缩的技术：
![](Pasted%20image%2020210704010154.png)
http://mitchgordon.me/machine/learning/2019/11/18//all-the-ways-to-compress-BERT.html
新出现的architecture，主要是用于让模型读更长的文章
![](Pasted%20image%2020210704010504.png)

## How to fine-tune
![](Pasted%20image%2020210704010714.png)
nlp任务的分类：
![](Pasted%20image%2020210704010734.png)
### input：
单个句子为输入，不用讲，这时默认情况
多个句子为输入：
在两个sentense之间加一个[SEP]，然后把整个一长串当作一个很长的句子送到模型中，输出embedding，就结束了
![](Pasted%20image%2020210704011004.png)

### output：
one class：
再训练的时候，在句首加一个特殊的token[CLS]，在看到[CLS]这个词的时候，就产生一个和整个句子有关的embedding。然后把这个特殊的embedding送入layer中。
![](Pasted%20image%2020210704011228.png)
或者另一种方法：
把所有token的embedding都都到后边的layer中
![](Pasted%20image%2020210704011331.png)
class for each token:
后面的layer给每个embedding一个class
![](Pasted%20image%2020210704011536.png)
copy from input:
![](Pasted%20image%2020210704011616.png)
常用于 extraction-based QA的任务，这种任务的输出是两个int，用来指示answer出现在文章中的开始和结束的位置

![](Pasted%20image%2020210704011912.png)
输入是question和document， 
后续的model，就只是两个vector。（也可以用lstm之类的其他设计）
其中一个用来侦测start的位置
![](Pasted%20image%2020210704012114.png)
另一个用来侦测end
![](Pasted%20image%2020210704012203.png)

General sequence：
第一种
可以接一个decoder，但是缺点是这个task specific的decoder没有pre-train过，效果可能不好
第二种
iteratively经过task specific产生sequence的下一个词汇（其实相当于把pre-trained model当作decoder来使用）
![](Pasted%20image%2020210704012623.png)

### fine-tune方法
第一种，只fine-tune task specific的部分
![](Pasted%20image%2020210704012808.png)
第二种，两部分一起学习
![](Pasted%20image%2020210704012843.png)

fine-tune整个model往往performance会更好，但是可能会遇到一些问题：
训练之后，我们可能为每个task都存一个巨大的model，耗费空间
![](Pasted%20image%2020210704013028.png)

adapter：
在pre-trained model中加入一些layer，叫做adapter。在训练的时候就只调这些layer，这样储存的时候就能只保存这个adapter了
![](Pasted%20image%2020210704013312.png)


Weighted features：
把不同layer的embedding作weighted sum，得到一个新的embedding
其中这个weight可以是学出来的
![](Pasted%20image%2020210704013810.png)

### Why pre-train models?
![](Pasted%20image%2020210704013929.png)
![](Pasted%20image%2020210704014004.png)


## How to pre-train?

早期：supervised，用翻译任务
![](Pasted%20image%2020210704135449.png)

现在：self-supervised
![](Pasted%20image%2020210704135717.png)


self-supervised中，如何制造输入跟输出的关系？
### 1. predict next token
![](Pasted%20image%2020210704140021.png)
注意每次输入，不能送入未来的文本
早期用lstm
![](Pasted%20image%2020210704140119.png)
后期用self-attention
![](Pasted%20image%2020210704140248.png)
注意self-attention要加约束，防止看到未来
但是可以发现这样的话，model 只看到了左边的文本，也就是说embedding是由左边的文本决定的
ELMO中也考虑到利用右边的文本，采用了bidirectional lstm，把正向的lstm得到的vector跟逆向得到的concat起来，用来最终表示一个word
![](Pasted%20image%2020210704141415.png)
但是这样的话，两个lstm是没有交汇的，每个lstm都只看到半个句子

BERT就可以解决这个问题
它把其中一个token盖住(或者随即换成另外一个token)。
然后来预测被盖起来的部分原来是什么
BERT中的self-attention是没有constraint的，每一个word都可以看到其他所有的word，因此可以看到整个句子
![](Pasted%20image%2020210704141801.png)
BERT的思路其实跟w2v中的CBOW是一样的，只是模型复杂程度不同
![](Pasted%20image%2020210704141836.png)

只盖住一个token的话，有的时候模型就不会关注很长的context，biruzhongwen，效果不太好。
所起可以盖住整个word或者prase或者entity
![](Pasted%20image%2020210704142341.png)
spanbert：一次盖住随机长度的部分
![](Pasted%20image%2020210704142510.png)
spanbert中还提出了一种训练的方法：span boundary objective
![](Pasted%20image%2020210704142716.png)
用MASK的左右两边的embedding预测中间的word

XLNet：
从language  model的观点来看：
 以往只能看到left context，但是在XLNet中会把word打乱
![](Pasted%20image%2020210711142620.png)
从bert的观点来看：
传统的bert先用mask挡住某一个word然后可以看到整个句子，而XLNet只能看到句子的一部分
![](Pasted%20image%2020210711142834.png)
另外在XLNet中没有mask这个token，不过还是要用positional encoding告诉model要预测的是哪个位置的word

***
BERT其实不擅长generative的任务：
对于language model，在训练的时候本来就是predict next token，因此生成的性能比较好

但对于bert，在训练的时候由于用的是mask，它看到的是左右两边的context，而在预测的时候只能看到左边的context，因此性能并不会很好(这里指的是autoregressive model，也就是在生成token的时候由左及右地生成token)(因此在non-autoregressive的bert中，也许就没有问题了)

这里先只讨论autoregressivve的情况：
bert因为不擅长生成，因此不适合直接作为seq2seq

其实可以直接pretrain一个encoder+decoder，训练时让输出等于输入
但要对input做一定程度的corrupt，防止模型直接原样输出，学不到东西
![](Pasted%20image%2020210711144102.png)
input coruption的方法：
MASS：
把input随机地mask起来
MASS只要求还原mask的部分，不需要原样输出其他的部分
![](Pasted%20image%2020210711144317.png)
BART：
除了mask以外，又提出其他的方法：
![](Pasted%20image%2020210711144606.png)

UniLM：
![](Pasted%20image%2020210711144752.png)
同时既是encoder又是decoder，又是seq2seq，而且是一个整体，不是encoder和decoder分开的结构
![](Pasted%20image%2020210711144900.png)


### 2. Replace or not?
ELECTRA只回答binary的问题，回答每个位置是否发生了替换
这里的“替换”，可以理解为“语法没有错，但语义稍微有点怪怪的”
![](Pasted%20image%2020210711145304.png)
![](Pasted%20image%2020210711145335.png)
优点：
* 比reconstriction要简单
* 每个position都能产生error

如何产生这种“怪怪的”input？
用另一个bert预测某个位置的mask是什么
然后把这个句子丢给ELECTRA，问模型知不知道有些词是bert产生的
![](Pasted%20image%2020210711145925.png)
不是GAN，因为小的bert不用骗过model，就自己train自己的
![](Pasted%20image%2020210711150120.png)
从上图可见ELECTRA的运算量比较小（虽然也还是自己train不了）


### 3. Sentense Level Embedding
之前讲的都是每一个word一个embedding，现在介绍给整个句子一个embedding的情况
![](Pasted%20image%2020210711160705.png)

思路：类似word的表示，通过邻近的句子来表示句子
skip thought：
输入一个句子，得到embedding，预测下一句。如果两个input的句子相似，就可能得到相似的embedding（训练困难）
quick thought：
与ELECTRA相似，转化为binary问题：
输入两个句子到encoder中得到两个embedding，然后根据similarity判断这两个句子是否相邻(NSP: Next Sentence prediction)
![](Pasted%20image%2020210711161101.png)
原始的BERT中，
之前说过CLS这个token用来输出整个句子相关的一个值。接收到CLS之后，模型会判断SEP前后的两个句子是不是相邻的。
![](Pasted%20image%2020210711165151.png)

RoBERTa和XLNet的paper中提到说NSP没有什么用

还有一种思路叫SOP：Sentence Order Prediction
把两个相邻的句子先后输入，输出True，把这两个句子颠倒一下，输出False
ALBERT中有用到SOP

structBERT(Alice)中用到类似SOP的方法

Google的文章T5比较了这些pretrain的性能：
![](Pasted%20image%2020210711165902.png)


BERT加上external knowledge的方法叫做ERNIE
![](Pasted%20image%2020210711170024.png)

