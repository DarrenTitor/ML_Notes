# lecture03-BNrepresentation
![](Pasted%20image%2020210524205405.png)
![](Pasted%20image%2020210524205438.png)

![](Pasted%20image%2020210524205459.png)




![](Pasted%20image%2020210524205519.png)
这里有一个很好的head to head的例子：
一个parent表示钟表是否坏掉了，另一个parent表示是否堵车，child表示是否迟到。未观测到child时两个parent是独立的，当观测到已经迟到时，两个parent则会“竞争”引起迟到的原因。当我们发现堵车发生了，那么很有可能钟表没有坏，反之亦然。

![](Pasted%20image%2020210524205640.png)



![](Pasted%20image%2020210524141033.png)
ancestral gragh：只保留node of interest和它们的祖先
moralize：把边变为无向边，然后连接coparent

另一种简单的方法，只需要原本的gragh，不需要moralize之类的：
**bayes ball**
![](Pasted%20image%2020210524211048.png)




![](Pasted%20image%2020210524143410.png)
![](Pasted%20image%2020210524143440.png)
和无向图的结论相似：
如果对着一个graph写出distribution P，graph中的independence一定会出现在P中。
如果我们从P中“拼凑出”一个independence，则不能保证这些变量在graph中可以d-seperate，因为可能只是数字上的巧合。
或者说，如果在graph中变量不能被d-seperate，不代表在P中就一定dependent

![](Pasted%20image%2020210524211133.png)

