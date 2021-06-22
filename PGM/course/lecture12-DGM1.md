# lecture12-DGM1

![](Pasted%20image%2020210606140113.png)
![](Pasted%20image%2020210606140142.png)
![](Pasted%20image%2020210606140232.png)
![](Pasted%20image%2020210606140249.png)


![](Pasted%20image%2020210606140316.png)
## Wake sleep algorithm
![](Pasted%20image%2020210606140438.png)
![](Pasted%20image%2020210606140448.png)
先回顾一下variational inference，可以看到VI中的Estep和Mstep都是对于free energy进行优化。

![](Pasted%20image%2020210606140613.png)
而在wake sleep algorithm中，可以看到KL中p和q颠倒了，这其实是为了计算上的方便。
wake sleep algorithm总体上训练两个单独的model，inference model和generate model，分别对应$p_{\theta}(x|z)$和$q_{\phi}(z|x)$.

![](Pasted%20image%2020210606141829.png)
在wake phase，我们想对于θ优化，观察free energy的形式可以看出，θ只出现在期望的object中，而期望对应的分布与θ无关，只与φ有关。因此我们可以直接在q上对于p进行采样，从而优化θ

![](Pasted%20image%2020210606142305.png)
而在sleep phase中，我们想要对φ优化，但是看到φ出现在了我们想要sample的distribution中，没法sample，我们只能learn这个φ。
而如果经过一些变换，
![](Pasted%20image%2020210606142535.png)
我们可以得到一个log P的形式。因为P可能是一个很小的值，logP就会很大。总体上logP的尺度变化就会很大，梯度的尺度变化也会很大，在实际操作中没法实现。

因此在wake sleep algorithm中，就把KL中的p和q调换了。换句话说，我们换了另一个loss function。

在调换之后，可以看到
![](Pasted%20image%2020210606143007.png)
我们同样可以使用sample了。尽管我们有observed X，我们还是ignore掉X。然后从之前generative model的p(x, z)中，'dream'出x和z，然后进行sample。
![](Pasted%20image%2020210606143227.png)


***
![](Pasted%20image%2020210606143559.png)






***
## VAE
![](Pasted%20image%2020210606150436.png)
![](Pasted%20image%2020210606150446.png)
我们对原本复杂的**梯度**进行reparameterization，直接对于梯度建立一个discrimitive的分布，重新设置这个分布的形式，然后去学习这个新的分布的参数。理论上这样可以解决原本variance过大的问题。
![](Pasted%20image%2020210606150425.png)
![](Pasted%20image%2020210606151348.png)


***
## GAN


![](Pasted%20image%2020210606151523.png)
在GAN中，我们只有generative model，没有inference model。
![](Pasted%20image%2020210606151911.png)
我们需要达到两个目标：
![](Pasted%20image%2020210606152029.png)
In practical，我们使用下面这个式子解决gradient vanishing的问题。
![](Pasted%20image%2020210606152115.png)

![](Pasted%20image%2020210606152355.png)

![](Pasted%20image%2020210606152446.png)



***
## A unified view of deep generative models

对于GAN，最普通的数学表示是这样的，然而这个跟之前的variational EM没有什么共同点。
![](Pasted%20image%2020210608145411.png)

![](Pasted%20image%2020210608164532.png)
注意此时在GAN两个step中，我们要优化的loss其实是不一样的，多了一个reverse。
![](Pasted%20image%2020210608173846.png)
**注意虽然GAN是用来生成图片x的，但真正observed的是label y。因此我们在这里把x当作latent，因此生成图片的过程是对于latent variable x的inference，而不是generate。**
注意variational EM，lower bound 可以有两种写法，之前推导一直用的是右上角的这种，现在写成另一种，以便观察和GAN的相似性。

![](Pasted%20image%2020210608191821.png)
在vEM中，x是observed的，z是latent的
在GAN中，y是observed，代表是真实data还是生成的data；x是latent的，用来表示data
可以看到最大的区别是，GAN中没有和prior的KL。因为GAN本身就没有prior，z是trivial的。




***
![](Pasted%20image%2020210611205243.png)
在GAN的更新过程中，我们可以把梯度写成KL减去一个JSD的形式。这个JSD我们这里不管，因为变不成其他的形式。
我们这里来观察KL的性质。

可以看到这个跟我们在VEM中想要做的事是类似的，都是让两个latent variable的posterior尽可能相近。GAN其实想让generative p和discrimitive的posterior q尽可能相近。

前几行用于说明$KL(p_{\theta}(x|y=1)||q^r(x|y=1))$是不用优化的。由带★号的式子可以看到，$q^r(x|y=0)$可以看作是真实data的分布和generated data的分布的mixture。
![](Pasted%20image%2020210611213225.png)
所以learning的过程其实是让$p_{g_{\theta}}(x|y)$靠近mixture，也就相当于靠近$p_{data}(x)$。

然后注意到KL的非对称性，可以得到以下的性质：
![](Pasted%20image%2020210611213643.png)
这一点在PMRL中已经讲过了，也就是p会“缩在”q的mode中。

![](Pasted%20image%2020210611213800.png)
而JSD是对称的，对此没有影响。
因此综合起来，GAN会丢失true distribution中的mode。

GAN的image是sharp的，就跟这个上面这个性质有关。GAN在learning的过程中会倾向于选择一部分更大的mode，而丢掉一些小mode。因此在生成style的任务中，加入输入10张Astyle，1张Bstyle，那么输出的image很可能不会来自B。
***
我们尝试用同样的思路来分析VAE，然而VAE中没有GAN中的indicator y。那我们就假设一个y，因为VAE中所有的图片都是生成的，因此所有图片的label y都是fake。
然后经过一系列推导可以得出下面的式子，可以看到也是一个KL的形式。但是与GAN不同的是，VAE对于latent的inference出现在KL的右边，因此VAE倾向于把所有的mode都包括进来。表现为VAE生成的图象比较模糊。
![](Pasted%20image%2020210611215704.png)


***
![](Pasted%20image%2020210611220139.png)
![](Pasted%20image%2020210611220329.png)
















