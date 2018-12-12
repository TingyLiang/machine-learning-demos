import tensorflow as tf
import os
import json


class ResourceUtil:

    def update_config(self, origin):
        '''高阶API调用时，重新生成 tf 的run config'''
        request = {
            "type": "tensorflow",
            "version": "1.5.0",
            "file": "/tmp/mnist.tgz",
            "data": "/test/data",
            "export": "s3://tmp",
            "resource": {
                "cpu": 2,
                "mem": "4g",
                "gpu": 1
            },
            "detail": {
                "ps": 1,
                "worker": 2
            }
        }

        cluster = {'chief': ['localhost:2222'],
                   'ps': ['localhost:2223', 'localhost:2224'],
                   'worker': ['localhost:2224', 'localhost:2225']}
        resource = self.get_resource(request)

        os.environ['TF_CONFIG'] = json.dumps(
            {'cluster': cluster,
             'task': {'type': 'worker', 'index': 1}
             })
        return tf.estimator.RunConfig(model_dir=origin.model_dir, tf_random_seed=origin.tf_random_seed,
                                      train_distribute=origin.train_distribute,
                                      save_summary_steps=origin.save_summary_steps,
                                      save_checkpoints_steps=origin.save_checkpoints_steps,
                                      save_checkpoints_secs=origin.save_checkpoints_secs,
                                      session_config=origin.session_config,
                                      keep_checkpoint_max=origin.keep_checkpoint_max,
                                      keep_checkpoint_every_n_hours=origin.keep_checkpoint_every_n_hours,
                                      log_step_count_steps=origin.log_step_count_steps,
                                      device_fn=origin.device_fn,
                                      eval_distribute=origin.eval_distribute,
                                      )

    def get_resource(self, request):
        # TODO 和 k8s服务对接，获取资源
        resource = {}
        return resource
