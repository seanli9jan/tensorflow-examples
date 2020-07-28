import tensorflow as tf
import os

def tf_disable_logging(interactive="DEBUG"):
    level = {"DEBUG":'0', "INFO":'1', "WARNING":'2', "ERROR":'3'}
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = level[interactive]

def tf_limit_gpu():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
      try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
          tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
      except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

def tf_save_model(obj, export_dir):
    tf.get_logger().setLevel("ERROR")
    obj.save(export_dir)
    tf.get_logger().setLevel("WARNING")

def tf_load_model(export_dir, custom_objects=None, compile=True):
    return tf.keras.models.load_model(export_dir, custom_objects=custom_objects, compile=compile)

def tf_print_tensor(x, message='', map_fn=None):
    def _func(x):
        return map_fn(x) if map_fn else x

    def _print(x):
        tf.print(message, _func(x)) if message else tf.print(_func(x))
        return x

    return tf.keras.layers.Lambda(_print)(x)
