import tensorflow as tf
import numpy as np
import os
import json
from ResourceUtil import ResourceUtil

'''
tf 使用 estimator 进行分布式训练:
存在几种不同的分布式策略
1.MirroredStrategy: This does in-graph replication with synchronous training on many GPUs on one machine.
 Essentially, we create copies of all variables in the model's layers on each device. 
 We then use all-reduce to combine gradients across the devices before applying them to the variables to keep them in sync.

2. CollectiveAllReduceStrategy: This is a version of MirroredStrategy for multi-worker training.   
It uses a collective op to do all-reduce. This supports between-graph communication and synchronization, 
and delegates the specifics of the all-reduce implementation to the runtime (as opposed to encoding it in the graph). 
This allows it to perform optimizations like batching and switch between plugins that support different hardware or algorithms.
In the future, this strategy will implement fault-tolerance to allow training to continue when there is worker failure.

3. ParameterServerStrategy: This strategy supports using parameter servers either for multi-GPU local training or asynchronous multi-machine training. 
 When used to train locally, variables are not mirrored, instead they are placed on the CPU and operations are replicated across all local GPUs. 
 In a multi-machine setting, some are designated as workers and some as parameter servers. 
 Each variable is placed on one parameter server. Computation operations are replicated across all GPUs of the workers.
'''


def model_fn(features, labels, mode):
    # 生成一个网络，方式为不断添加层
    net = tf.layers.Dense(16, activation='relu', input_shape=(10,))
    net = tf.layers.Dense(1, activation='sigmoid')
    sigmod = net(features)
    loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=[float(32), float(2)], logits=[float(32), float(2)])

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {"sigmoid": sigmod}
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)
    # loss = tf.losses.mean_squared_error(
    #     labels=labels, predictions=tf.reshape(sigmod, [None, 1]))
    # loss = None

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode)
        # return tf.estimator.EstimatorSpec(mode, loss=loss)

    if mode == tf.estimator.ModeKeys.TRAIN:
        # train_op = tf.train.GradientDescentOptimizer(0.2).get_name()
        train_op = tf.train.GradientDescentOptimizer(0.2).minimize(loss)
        return tf.estimator.EstimatorSpec(mode, train_op=train_op)


def input_fn():
    x = np.random.random((1024, 10))
    y = x ** 2 + 5
    x = tf.cast(x, tf.float32)
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.repeat(10)
    dataset = dataset.batch(32)
    return dataset


def eval_input_fn():
    input_fn()


def model_main():
    # 该策略支持图间异步训练，支持不同硬件
    distribution = tf.contrib.distribute.CollectiveAllReduceStrategy(
        num_gpus_per_worker=2)
    config = tf.estimator.RunConfig(train_distribute=distribution)
    config = ResourceUtil().update_config(config)

    estimator = tf.estimator.Estimator(model_fn=model_fn, config=config, model_dir='../output/cluster')

    print(config.cluster_spec)
    print("num_ps_replicas:" + str(config.num_ps_replicas))
    print('master:' + config.master)
    print(config.train_distribute)
    print('task type: ' + config.task_type)
    print(config.task_id)

    train_spec = tf.estimator.TrainSpec(input_fn=input_fn)
    eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn)

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


# # 高阶API下设置集群，需要在环境变量中进行配置
# cluster = {'chief': ['localhost:2222'],
#            'ps': ['localhost:2223', 'localhost:2224'],
#            'worker': ['localhost:2224', 'localhost:2225']}
# os.environ['TF_CONFIG'] = json.dumps(
#     {'cluster': cluster,
#      'task': {'type': 'worker', 'index': 0}
#      })

model_main()
