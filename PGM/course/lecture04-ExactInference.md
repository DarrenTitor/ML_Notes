# lecture04-ExactInference

![](Pasted%20image%2020210524214810.png)
![](Pasted%20image%2020210525104734.png)
![](Pasted%20image%2020210525104756.png)
![](Pasted%20image%2020210525104820.png)
![](Pasted%20image%2020210525104844.png)


![](Pasted%20image%2020210525104901.png)


我们在eliminate某一个变量a的时候，先把所有与a相关的term放到一起，然后把这些term对a进行summation，就得到了一个关于“剩余term”的function m(b,c)，我们称之为message
![](Pasted%20image%2020210524224307.png)
![](Pasted%20image%2020210524225104.png)
![](Pasted%20image%2020210524225409.png)


Variable Elimination ：
我们可以把上面的过程推广到graph，就是Variable Elimination和sum product
![](Pasted%20image%2020210525105335.png)
![](Pasted%20image%2020210525105503.png)
![](Pasted%20image%2020210525105536.png)
![](Pasted%20image%2020210525105608.png)
![](Pasted%20image%2020210525105711.png)
。。。
![](Pasted%20image%2020210525105813.png)

![](Pasted%20image%2020210525105904.png)
sum-product的复杂度取决于clique tree中最大的clique。

![](Pasted%20image%2020210525110016.png)
![](Pasted%20image%2020210525110043.png)
![](Pasted%20image%2020210525110114.png)
![](Pasted%20image%2020210525110127.png)
![](Pasted%20image%2020210525110209.png)
![](Pasted%20image%2020210525110233.png)


![](Pasted%20image%2020210525110251.png)
clique只跟elimination的顺序有关，而与query无关。因此我们在选定一个elimination的顺序之后，就可以把message存起来，可以重复使用。
![](Pasted%20image%2020210525110308.png)
![](Pasted%20image%2020210525110318.png)
![](Pasted%20image%2020210525110332.png)
![](Pasted%20image%2020210525110345.png)

![](Pasted%20image%2020210525110457.png)