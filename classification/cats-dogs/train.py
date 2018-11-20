from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_integer(flag_name='batch_size', default_value=16, docstring='Batch 大小')
flags.DEFINE_string(flag_name='data_dir', default_value='../../output/catsVSdogs/data', docstring='数据存放位置')
flags.DEFINE_string(flag_name='model_dir', default_value='../../output/catsVSdogs/model', docstring='模型存放位置')
flags.DEFINE_integer(flag_name='steps', default_value=1000, docstring='训练步数')
flags.DEFINE_integer(flag_name='classes', default_value=2, docstring='类别数量')
FLAGS = flags.FLAGS

MODES = [tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL, tf.estimator.ModeKeys.PREDICT]


def input_fn(mode, batch_size=1):
    """输入函数"""

    def parser(serialized_example):
        """如何处理数据集中的每一个数据"""

        # 解析单个example对象
        features = tf.parse_single_example(
            serialized_example,
            features={
                'image/height': tf.FixedLenFeature([], tf.int64),
                'image/width': tf.FixedLenFeature([], tf.int64),
                'image/depth': tf.FixedLenFeature([], tf.int64),
                'image/encoded': tf.FixedLenFeature([], tf.string),
                'image/class/label': tf.FixedLenFeature([], tf.int64),
            })

        # 获取参数
        height = tf.cast(features['image/height'], tf.int32)
        width = tf.cast(features['image/width'], tf.int32)
        depth = tf.cast(features['image/depth'], tf.int32)

        # 还原image
        image = tf.decode_raw(features['image/encoded'], tf.float32)
        image = tf.reshape(image, [height, width, depth])
        image = image - 0.5

        # 还原label
        label = tf.cast(features['image/class/label'], tf.int32)

        return image, tf.one_hot(label, FLAGS.classes)

    if mode in MODES:
        tfrecords_file = os.path.join(FLAGS.data_dir, mode + '.tfrecords')
    else:
        raise ValueError("Mode 未知")

    assert tf.gfile.Exists(tfrecords_file), ('TFRrecords 文件不存在')

    # 创建数据集
    dataset = tf.data.TFRecordDataset([tfrecords_file])
    # 创建映射
    dataset = dataset.map(parser, num_parallel_calls=1)
    # 设置batch
    dataset = dataset.batch(batch_size)
    # 如果是训练，那么就永久循环下去
    if mode == tf.estimator.ModeKeys.TRAIN:
        dataset = dataset.repeat()
    # 创建迭代器
    iterator = dataset.make_one_shot_iterator()
    # 获取 feature 和 label
    images, labels = iterator.get_next()

    return images, labels


def my_model(inputs, mode):
    """写一个网络"""
    net = tf.reshape(inputs, [-1, 224, 224, 1])
    net = tf.layers.conv2d(net, 32, [3, 3], padding='same', activation=tf.nn.relu)
    net = tf.layers.max_pooling2d(net, [2, 2], strides=2)
    net = tf.layers.conv2d(net, 32, [3, 3], padding='same', activation=tf.nn.relu)
    net = tf.layers.max_pooling2d(net, [2, 2], strides=2)
    net = tf.layers.conv2d(net, 64, [3, 3], padding='same', activation=tf.nn.relu)
    net = tf.layers.conv2d(net, 64, [3, 3], padding='same', activation=tf.nn.relu)
    net = tf.layers.max_pooling2d(net, [2, 2], strides=2)
    # print(net)
    net = tf.reshape(net, [-1, 28 * 28 * 64])
    net = tf.layers.dense(net, 1024, activation=tf.nn.relu)
    net = tf.layers.dropout(net, 0.4, training=(mode == tf.estimator.ModeKeys.TRAIN))
    net = tf.layers.dense(net, FLAGS.classes)
    return net


def my_model_fn(features, labels, mode):
    """模型函数"""

    # 可视化输入
    tf.summary.image('images', features)

    # 创建网络
    logits = my_model(features, mode)

    predictions = {
        'classes': tf.argmax(input=logits, axis=1),
        'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
    }

    # 如果是PREDICT，那么只需要predictions就够了
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # 创建Loss
    loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits, scope='loss')
    tf.summary.scalar('train_loss', loss)

    # 设置如何训练
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
        train_op = optimizer.minimize(loss, tf.train.get_or_create_global_step())
    else:
        train_op = None

    # 获取训练精度
    accuracy = tf.metrics.accuracy(
        tf.argmax(labels, axis=1), predictions['classes'],
        name='accuracy')

    accuracy_topk = tf.metrics.mean(
        tf.nn.in_top_k(predictions['probabilities'], tf.argmax(labels, axis=1), 2),
        name='accuracy_topk')

    metrics = {
        'test_accuracy': accuracy,
        'test_accuracy_topk': accuracy_topk
    }

    # 可视化训练精度
    tf.summary.scalar('train_accuracy', accuracy[1])
    tf.summary.scalar('train_accuracy_topk', accuracy_topk[1])

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=metrics)


def main(_):
    # 监视器
    logging_hook = tf.train.LoggingTensorHook(
        every_n_iter=100,
        tensors={
            'accuracy': 'accuracy/value',
            'accuracy_topk': 'accuracy_topk/value',
            'loss': 'loss/value'
        },
    )

    # 创建 Estimator
    model = tf.estimator.Estimator(
        model_fn=my_model_fn,
        model_dir=FLAGS.model_dir)

    for i in range(20):
        # 训练
        model.train(
            input_fn=lambda: input_fn(tf.estimator.ModeKeys.TRAIN, FLAGS.batch_size),
            steps=FLAGS.steps,
            hooks=[logging_hook])

        # 测试并输出结果
        print("=" * 10, "Testing", "=" * 10)
        eval_results = model.evaluate(
            input_fn=lambda: input_fn(tf.estimator.ModeKeys.EVAL))
        print('Evaluation results:\n\t{}'.format(eval_results))
        print("=" * 30)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
