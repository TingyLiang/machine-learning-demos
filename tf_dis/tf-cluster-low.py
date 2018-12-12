import argparse
import sys
import tensorflow as tf
import math
from tensorflow.examples.tutorials.mnist import input_data
import time

''' tf 分布式训练低层API实现示例代码，使用图间，异步训练模式'''
FLAGS = None
IMAGE_PIXELS = 28


def main(_):
    # 以下部分代码是集群创建
    ps_hosts = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")

    # Create a cluster from the parameter server and worker hosts.
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

    # Create and start a server for the local task.
    server = tf.train.Server(cluster,
                             job_name=FLAGS.job_name,
                             task_index=FLAGS.task_index)

    mnist = input_data.read_data_sets("../data/mnist", one_hot=True)
    if FLAGS.job_name == "ps":
        server.join()
    elif FLAGS.job_name == "worker":

        # Assigns ops to the local worker by default.
        with tf.device(tf.train.replica_device_setter(
                worker_device="/job:worker/task:%d" % FLAGS.task_index,
                cluster=cluster)):
            # 实际只有这部分代码是真正的核心部分公用代码
            # ==============Build model...
            # Variables of the hidden layer
            # 定义隐藏层参数变量，这里是全连接神经网络隐藏层
            hid_w = tf.Variable(
                tf.truncated_normal(
                    [IMAGE_PIXELS * IMAGE_PIXELS, 100],
                    stddev=1.0 / IMAGE_PIXELS),
                name="hid_w")
            hid_b = tf.Variable(tf.zeros([100]), name="hid_b")
            # Variables of the softmax layer
            # 定义Softmax 回归层参数变量
            sm_w = tf.Variable(
                tf.truncated_normal(
                    [100, 10],
                    stddev=1.0 / math.sqrt(100)),
                name="sm_w")
            sm_b = tf.Variable(tf.zeros([10]), name="sm_b")
            # Ops: located on the worker specified with FLAGS.task_index
            # 定义模型输入数据变量
            x = tf.placeholder(tf.float32, [None, None])
            y_ = tf.placeholder(tf.float32, [None, 10])
            # 构建隐藏层
            hid_lin = tf.nn.xw_plus_b(x, hid_w, hid_b)
            hid = tf.nn.relu(hid_lin)
            # 构建损失函数和优化器
            y = tf.nn.softmax(tf.nn.xw_plus_b(hid, sm_w, sm_b))
            cross_entropy = -tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))
            # loss function :cross_entropy
            loss = -tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))

            global_step = tf.train.get_or_create_global_step()

            train_op = tf.train.AdagradOptimizer(0.01).minimize(
                loss, global_step=global_step)
            # =====================end defining model

        # The StopAtStepHook handles stopping after running given steps.
        hooks = [tf.train.StopAtStepHook(last_step=1000)]

        # The MonitoredTrainingSession takes care of session initialization,
        # restoring from a checkpoint, saving to a checkpoint, and closing when done
        # or an error occurs.
        with tf.train.MonitoredTrainingSession(master=server.target,
                                               is_chief=(FLAGS.task_index == 0),
                                               checkpoint_dir="../output/train_logs",
                                               hooks=hooks) as mon_sess:
            while not mon_sess.should_stop():
                # Run a training step asynchronously.
                # See <a href="./../api_docs/python/tf/train/SyncReplicasOptimizer"><code>tf.train.SyncReplicasOptimizer</code></a> for additional details on how to
                # perform *synchronous* training.
                # mon_sess.run handles AbortedError in case of preempted PS.
                local_step = 0
                while True:
                    # Training feed
                    # 读入MNIST训练数据，默认每批次100张图片
                    batch_xs, batch_ys = mnist.train.next_batch(100)
                    train_feed = {x: batch_xs, y_: batch_ys}
                    _, step = mon_sess.run(train_op, train_feed)
                    local_step += 1
                    if local_step > 200:
                        break
                # Validation feed
                # 读入MNIST验证数据，计算验证的交叉熵
                # val_feed = {x: mnist.validation.images, y_: mnist.validation.labels}
                # val_xent = mon_sess.run(cross_entropy, feed_dict=val_feed)
                # print("After training , validation cross entropy = %g" %
                #       val_xent)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    # Flags for defining the tf.train.ClusterSpec
    parser.add_argument(
        "--ps_hosts",
        type=str,
        default="",
        help="Comma-separated list of hostname:port pairs"
    )
    parser.add_argument(
        "--worker_hosts",
        type=str,
        default="",
        help="Comma-separated list of hostname:port pairs"
    )
    parser.add_argument(
        "--job_name",
        type=str,
        default="",
        help="One of 'ps', 'worker'"
    )
    # Flags for defining the tf.train.Server
    parser.add_argument(
        "--task_index",
        type=int,
        default=0,
        help="Index of task within the job"
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

# On localhost:

# --ps_hosts=localhost:2222  --worker_hosts=localhost:2223,localhost:2224  --job_name=ps --task_index=0

# $ python trainer.py \
# --ps_hosts=localhost:2222  --worker_hosts=localhost:2223,localhost:2224   --job_name=worker --task_index=0

# $ python trainer.py \
#      --ps_hosts=localhost:2222 \
#      --worker_hosts=localhost:2223,localhost:2224 \
#      --job_name=worker --task_index=1
