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

np.pad(array, pad_width, mode='constant', **kwargs)#（待填充的数组，每个维度填充的长度，填充的类型，接受的字典-只有一些类型需要额外参数的时候使用）
                                                  #类型：
                                                  #constant（常数值填充，**kwargs 参数constant_values=(before, after)）
                                                  #edge（复制边缘值）
                                                  #linear_ramp（线性填充，**kwargs 参数end_values=(before, after)）
                                                  #maximum（边缘最大值填充）
                                                  #mean（边缘平均值填充）
                                                  #median（边缘中位数填充）
                                                  #minimum（边缘最小值）
                                                  #reflect（反射填充，不包含边缘）
                                                  #symmetric（对称填充，包含边缘）
                                                  #wrap（环绕填充，数组首尾相连）

numpy.lib.stride_tricks.as_strided(x, shape=None, strides=None, subok=False, writeable=True)
#x:原始输入数组
#shape:新视图的形状
#strides:新视图中每个维度的步长（单位字节），决定窗口滑动的方式
#subok:是否返回子视图
#writeable:返回视图是不是可写


```
