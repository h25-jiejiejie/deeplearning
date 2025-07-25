```python
np.random.permutation(A,axis=)    #打乱顺序，返回新的排列,并且A可以是一个值代表从0开始的A个数量的排列
np.random.shuffled(A)          #直接打乱

np.random.seed(num)    #num是种子
np.random.rand(shape)  #随机生成0到1之间的数值（平均）
np.random.randn(shape)#随机正态分布生成R上的数值，概率符合正太分布
np.random.linspace(star,end,num)#均匀分布num个值

np.linalg.norm(A,axis,keepdim)#将指定维度上的值进行平均和的开放
np.dot(A,B)#矩阵乘法
np.sum(A,axis,keepdim)#和
np.outer(A,B)#会将两个矩阵变成一维向量，然后将A的第i个元素和B的j个元素进行相乘得到新的值放进矩阵的新地方，输出Anum,Bnum的矩阵
np.mutiply(A,B)#矩阵下标相同的值进行相乘

```
