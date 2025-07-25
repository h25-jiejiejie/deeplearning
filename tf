```python
注意：张量tensor类型是tf.Tensor(data,shape，dtype),如果想只拿到数据就要张量后面加.numpy()
    很多在np中的操作这里都有，比如初始化1和0啥的，加减

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
    dtype
    name
)

tf.nn.relu(features, name=None)#激活函数

模型定义：
tf.keras.Model
你需要定义一个子类，继承自 tf.keras.Model，然后重写 __init__ 和 call 方法。
__init__：在这里定义网络的层，例如卷积层、全连接层等。
call：在这里定义前向传播的逻辑，通常是逐层传递数据。
编译和训练：与 Sequential 模型类似，你可以使用 compile 方法指定优化器、损失函数和评估指标，并使用 fit 进行训练。
# 1. 子类化 tf.keras.Model
class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # 定义层
        self.flatten = layers.Flatten(input_shape=(28, 28))  # 将输入图像展平
        self.dense1 = layers.Dense(128, activation='relu')   # 隐藏层，128 个神经元
        self.dense2 = layers.Dense(10, activation='softmax')  # 输出层，10 个神经元，softmax 激活函数

    def call(self, inputs):
        # 前向传播逻辑
        x = self.flatten(inputs)
        x = self.dense1(x)
        return self.dense2(x)

# 2. 创建模型实例
model = MyModel()

# 3. 编译模型
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 4. 加载数据
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

# 5. 预处理数据
X_train, X_test = X_train / 255.0, X_test / 255.0  # 归一化

# 6. 训练模型
model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))

# 7. 评估模型
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")



```
