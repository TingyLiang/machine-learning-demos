
import tensorflow as tf
from tensorflow import keras
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
'''tf keras 高层API单机模式示例代码'''
data = np.random.random((1000, 32))
labels = np.random.random((1000, 10))

val_data = np.random.random((100, 32))
val_labels = np.random.random((100, 10))

# 定义常量，用于创建数据流图
flags = tf.app.flags
# 因网络问题，这里将数据手动下载到项目指定目录下
flags.DEFINE_string("data_dir", "../data/mnist",
                    "Directory for storing mnist data")
FLAGS = flags.FLAGS

# 构建一个简单的全连接网络
model = keras.Sequential()
# Adds a densely-connected layer with 64 units to the model:
# tf.keras.layers  用于构建一层
# activation：设置层的激活函数。此参数由内置函数的名称指定，或指定为可调用对象。默认情况下，系统不会应用任何激活函数。
# kernel_initializer 和 bias_initializer：创建层权重（核和偏差）的初始化方案。此参数是一个名称或可调用对象，默认为 "Glorot uniform" 初始化器。
# kernel_regularizer 和 bias_regularizer：应用层权重（核和偏差）的正则化方案，例如 L1 或 L2 正则化。默认情况下，系统不会应用正则化函数。
model.add(keras.layers.Dense(64, activation='relu'))
# Add another:
model.add(keras.layers.Dense(64, activation='relu'))
# Add a softmax layer with 10 output units:
model.add(keras.layers.Dense(10, activation='softmax'))

# 配置模型的学习流程
model.compile(optimizer=tf.train.AdamOptimizer(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 模型训练，设置训练集，迭代次数，测试集等参数
model.fit(x=data, y=labels, epochs=10, validation_data=(val_data, val_labels), steps_per_epoch=3, batch_size=32)

# 保存模型至h5文件
model.save("my_model.h5")
keras.models.load_model("my_model.h5")
# 使用模型进行评估和预测
# model.evaluate()
# model.predict()
