from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from skimage import io, transform, color, util

flags = tf.flags
flags.DEFINE_string(flag_name='directory', default_value='F:/data/cats vs dogs/train', docstring='数据地址')
flags.DEFINE_string(flag_name='save_dir', default_value='../../output/catsVSdogs/model', docstring='保存地址')
flags.DEFINE_integer(flag_name='test_size', default_value=350, docstring='测试集大小')
FLAGS = flags.FLAGS

MODES = [tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL, tf.estimator.ModeKeys.PREDICT]


def _float_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.FloatList(value=value))


def _int_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def convert_to_tfrecord(mode, anno):
    """转换为TfRecord"""

    assert mode in MODES, "模式错误"

    filename = os.path.join(FLAGS.save_dir, mode + '.tfrecords')

    with tf.python_io.TFRecordWriter(filename) as writer:
        for fnm, cls in tqdm(anno):

            # 读取图片、转换
            img = io.imread(fnm)
            img = color.rgb2gray(img)
            img = transform.resize(img, [224, 224])

            # 获取转换后的信息
            if 3 == img.ndim:
                rows, cols, depth = img.shape
            else:
                rows, cols = img.shape
                depth = 1

            # 创建Example对象
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        'image/height': _int_feature(rows),
                        'image/width': _int_feature(cols),
                        'image/depth': _int_feature(depth),
                        'image/class/label': _int_feature(cls),
                        'image/encoded': _bytes_feature(img.astype(np.float32).tobytes())
                    }
                )
            )
            # 序列化并保存
            writer.write(example.SerializeToString())


def get_folder_name(folder):
    """不递归，获取特定文件夹下所有文件夹名"""

    fs = os.listdir(folder)
    fs = [x for x in fs if os.path.isdir(os.path.join(folder, x))]
    return sorted(fs)


def get_file_name(folder):
    """不递归，获取特定文件夹下所有文件名"""

    fs = os.listdir(folder)
    fs = map(lambda x: os.path.join(folder, x), fs)
    fs = [x for x in fs if os.path.isfile(x)]
    return fs


def get_annotations(directory, classes):
    """获取所有图片路径和标签"""

    files = []
    labels = []

    for ith, val in enumerate(classes):
        fi = get_file_name(os.path.join(directory, val))
        files.extend(fi)
        labels.extend([ith] * len(fi))

    assert len(files) == len(labels), "图片和标签数量不等"

    # 将图片路径和标签拼合在一起
    annotation = [x for x in zip(files, labels)]

    # 随机打乱
    random.shuffle(annotation)

    return annotation


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


def main(_):
    class_names = get_folder_name(FLAGS.directory)
    annotation = get_annotations(FLAGS.directory, class_names)

    convert_to_tfrecord(tf.estimator.ModeKeys.TRAIN, annotation[FLAGS.test_size:])
    convert_to_tfrecord(tf.estimator.ModeKeys.EVAL, annotation[:FLAGS.test_size])


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
