from __future__ import print_function

import sys
import threading

# This is a placeholder for a Google-internal import.

from grpc.beta import implementations
import numpy
import tensorflow as tf

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
from tensorflow.examples.tutorials.mnist import input_data


tf.app.flags.DEFINE_integer('concurrency', 1,
                            'maximum number of concurrent inference requests')
tf.app.flags.DEFINE_integer('num_tests', 100, 'Number of test images')
tf.app.flags.DEFINE_string('server', 'localhost:9000', 'PredictionService host:port')
tf.app.flags.DEFINE_string('work_dir', '../../Data/MNIST_data/', 'Working directory. ')
FLAGS = tf.app.flags.FLAGS


def do_inference(hostport, work_dir, concurrency, num_test):
    mnist = input_data.read_data_sets(work_dir,one_hot=True)
    test_data = mnist.test
    host, port = hostport.split(":")
    channel = implementations.insecure_channel(host,int(port))
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

if __name__=="__main__":
    print(FLAGS.num_tests)
    do_inference(FLAGS.server, FLAGS.work_dir, FLAGS.concurrency, FLAGS.num_tests)
