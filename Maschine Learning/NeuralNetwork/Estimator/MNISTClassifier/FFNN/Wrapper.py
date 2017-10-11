import tensorflow as tf

# Wrapper for DNNClassifier. Otherwise can`t initialize ServingInputReceiver with float input
class Wrapper(tf.estimator.Estimator):
    def __init__(self, **kwargs):
        dnn = tf.estimator.DNNClassifier(**kwargs)

        def model_fn(mode, features, labels):
            spec = dnn._call_model_fn(features, labels, mode)

            export_outputs = None
            if mode=="infer":
                export_outputs = {
                    "serving_default": tf.estimator.export.PredictOutput(
                        {"scores": tf.identity(spec.export_outputs["serving_default"].scores,"output"),
                         "class": tf.identity(spec.export_outputs["serving_default"].classes,"class")})}

            # Replace export_outputs, took a String(tf.Example) as input. Now takes float[]
            copy = list(spec)
            copy[4] = export_outputs
            return tf.estimator.EstimatorSpec(mode, *copy)

        super(Wrapper, self).__init__(model_fn, kwargs["model_dir"], dnn.config)