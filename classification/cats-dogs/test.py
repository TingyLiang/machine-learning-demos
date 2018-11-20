"""Run inference a DeepLab v3 model using tf.estimator API."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import tensorflow as tf
import train
from skimage import io, transform, color, util

mode = tf.estimator.ModeKeys.PREDICT
_NUM_CLASSES = 2
image_size = [224, 224]
image_files = '/home/a/Datasets/cat&dog/test/44.jpg'
model_dir = './cat&dog_model/'


def main(unused_argv):
    # Using the Winograd non-fused algorithms provides a small performance boost.
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
    #
    model = tf.estimator.Estimator(
        model_fn=train.my_model_fn,
        model_dir=model_dir)

    def predict_input_fn(image_path):
        img = io.imread(image_path)
        img = color.rgb2gray(img)
        img = transform.resize(img, [224, 224])
        image = img - 0.5
        # preprocess image: scale pixel values from 0-255 to 0-1
        images = tf.image.convert_image_dtype(image, dtype=tf.float32)
        dataset = tf.data.Dataset.from_tensors((images,))
        return dataset.batch(1).make_one_shot_iterator().get_next()

    def predict(image_path):

        result = model.predict(input_fn=lambda: predict_input_fn(image_path=image_path))
        for r in result:
            print(r)
            if r['classes'] == 1:
                print('dog', r['probabilities'][1])
            else:
                print('cat', r['probabilities'][0])

    predict(image_files)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main=main)
