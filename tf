```python
注意：张量tensor类型是tf.Tensor(data,shape，dtype),如果想只拿到数据就要张量后面加.numpy()

tf.constant(value, dtype=None, shape=None, name='Const')#方法用来创建一个不可变的张量，用于表示常量（比如输入数据、超参数、固定权重等）
tf.Variable(initial_value, dtype=None, trainable=True, name=None)#可变的张量（tensor），它的值在训练过程中可以被更新，常用于存储神经网络的权重和偏置。trainable表示是否在optimizer.apply_gradients被更新（默认是true）
tf.mutmul(A,B)#矩阵乘法，支持3D（[batch,x,y]）前面的batch代表这个训练批次

tf.transpose(
    a,       # 要转置的张量
    perm=None,  # 指定转置后的维度顺序，列表或元组，比如 [1,0] 表示交换第0和第1维
    conjugate=False,  # 是否对复数做共轭转置，默认False
    name=None  # 操作名，可选
)
loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits,name=name)#2分类交叉熵损失，用于计算激活函数是sigmoid情况下。输出的是所有数据的损失的张量
mean_loss = tf.reduce_mean(input_tensor,axis,keepdim,name)#计算平均损失输出的是损失平均的张量

tf.one_hot(                                    #one-hot编码用于类别的分类，可以多类别一个东西，对应的输出上相应位置就是1（默认）
    indices,   # 输入的类别索引，通常是一个整数张量
    depth,     # 类别总数，指定编码的长度
    on_value=1,   # one-hot 编码中 "True" 的值，默认为 1
    off_value=0,  # one-hot 编码中 "False" 的值，默认为 0
    axis=-1     # one-hot 编码的轴，默认为最后一维            axis=-1 也表示将 one-hot 编码应用于张量的最后一维。
)
```
