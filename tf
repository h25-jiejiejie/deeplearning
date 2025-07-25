```python
tf.constant(value, dtype=None, shape=None, name='Const')#方法用来创建一个不可变的张量，用于表示常量（比如输入数据、超参数、固定权重等）
tf.Variable(initial_value, dtype=None, trainable=True, name=None)#可变的张量（tensor），它的值在训练过程中可以被更新，常用于存储神经网络的权重和偏置。trainable表示是否在optimizer.apply_gradients被更新（默认是true）
tf.mutmul(A,B)#矩阵乘法，支持3D（[batch,x,y]）前面的batch代表这个训练批次
tf.transpose(
    a,       # 要转置的张量
    perm=None,  # 指定转置后的维度顺序，列表或元组，比如 [1,0] 表示交换第0和第1维
    conjugate=False,  # 是否对复数做共轭转置，默认False
    name=None  # 操作名，可选
)
```
