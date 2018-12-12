import tensorflow as tf
from tensorflow import keras
from tensorflow.python.training import server_lib
import numpy as np
import os
import json

# from lenovotf import lestimator

''' tf 高层api分布式训练示例代码'''
# 定义一个简单模型
model = keras.Sequential()
model.add(keras.layers.Dense(16, activation='relu', input_shape=(10,)))
model.add(keras.layers.Dense(1, activation='sigmoid'))

optimizer = tf.train.GradientDescentOptimizer(0.2)

model.compile(loss='binary_crossentropy', optimizer=optimizer)


# 1.12 版本仅mirrorstrategy 支持模型summary
# model.summary()


# os.environ['TF_CONFIG'] = json.dumps(
#     {'cluster': cluster,
#      'task': {'type': 'worker', 'index': 0}})
# os.environ['TF_CONFIG'] = json.dumps(
#     {'cluster': cluster,
#      'task': {'type': 'chief', 'index': 0}})
# Example of evaluator node (evaluator is not part of training cluster):
# os.environ['TF_CONFIG'] = json.dumps(
#     {'cluster': cluster,
#      'task': {'type': 'evaluator', 'index': 0}})
# 定义输入管道。input_fn 会返回 tf.data.Dataset 对象，此对象用于将数据分布在多台设备上，每台设备处理输入批次数据的一部分。
def input_fn():
    x = np.random.random((1024, 10))
    y = np.random.randint(2, size=(1024, 1))
    x = tf.cast(x, tf.float32)
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.repeat(10)
    dataset = dataset.batch(32)
    return dataset


# 目前，tf.contrib.distribute.MirroredStrategy 是唯一受支持的分布策略。
# MirroredStrategy 通过在一台机器上使用规约在同步训练中进行图内复制。
# 要将 DistributionStrategy 与 Keras 搭配使用，
# 将keras模型转换为 tf.estimator.Estimator 对象

# 高阶API下设置集群，需要在环境变量中进行配置,必须启动chief节点，才能开始训练，否则
cluster = {'chief': ['localhost:2222'],
           'ps': ['localhost:2223', 'localhost:2224'],
           'worker': ['localhost:2224', 'localhost:2225']}
# os.environ['TF_CONFIG'] = json.dumps(
#     {'cluster': cluster,
#      'task': {'type': 'chief', 'index': 0}
#      })

# 该策略仅支持单机多CPU分布式训练
# strategy = tf.contrib.distribute.MirroredStrategy()
strategy = tf.contrib.distribute.CollectiveAllReduceStrategy()
config = tf.estimator.RunConfig(train_distribute=strategy)
# config = ResourceUtil().update_config(config)
print(config.cluster_spec)
print("num_ps_replicas:" + str(config.num_ps_replicas))
print('master:' + config.master)
print(config.train_distribute)
print('task type: ' + config.task_type)
print(config.task_id)

cluster = {'chief': ['localhost:2222'],
           'worker': ['localhost:2224', 'localhost:2225']}
# config.cluster_spec = cluster
# print('after modify: '+config.cluster_spec)

estimator = keras.estimator.model_to_estimator(keras_model=model, config=config,
                                               model_dir='F:/PythonWorkspace/machine-learning-demos/output/model_dir')  # return a tf estimator
train_spec = tf.estimator.TrainSpec(input_fn=input_fn)
eval_spec = tf.estimator.EvalSpec(input_fn=input_fn)

# 训练API调用的方式有两种

#  方式1 ,反复调用此脚本，在没有清空checkpoint文件的情况下会进行增量训练
tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
# NOTE: test fail
# lestimator.train_and_evaluate(estimator, train_spec=train_spec, eval_spec=eval_spec)

# 方式2
# estimator.train(input_fn)
# estimator.evaluate(input_fn)
# test pass
# lestimator.train(estimator, input_fn)
# lestimator.evaluate(estimator, input_fn)

model.save("c-model.h5")
# estimator.train(input_fn, steps=100)
