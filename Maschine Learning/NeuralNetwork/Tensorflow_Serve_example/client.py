from grpc.beta import implementations
import numpy
import sys
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2


tf.app.flags.DEFINE_integer('num_tests', 100, 'Number of test images')
tf.app.flags.DEFINE_string('server', 'localhost:9000', 'PredictionService host:port')
tf.app.flags.DEFINE_string('work_dir', 'export', 'Working directory. ')
FLAGS = tf.app.flags.FLAGS

def create_accept_response(label):
    def accept_response(result_future):
        response = numpy.array(result_future.result().outputs['scores'].float_val)
        prediction = numpy.argmax(response)
        sys.stdout.write('Server response prediction: '+str(prediction))
        sys.stdout.write('\nCorrect label is : '+str(numpy.argmax(label)))
    return accept_response

def do_inference(dataFrame):
    host, port = FLAGS.server.split(':')
    channel = implementations.insecure_channel(host, int(port))
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'mnist'
    request.model_spec.signature_name = 'predict_images'
    image, label = dataFrame
    request.inputs['images'].CopyFrom(
        tf.contrib.util.make_tensor_proto(image[0], shape=[1, image[0].size]))

    result_future = stub.Predict.future(request, 5.0)  # 5 seconds
    result_future.add_done_callback(create_accept_response(label))


if __name__=="__main__":
    mnist = input_data.read_data_sets("../../Data/MNIST_data",one_hot=True)
    do_inference(mnist.test.next_batch(1))