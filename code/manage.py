import tensorflow as tf
from keras import backend as K

# Manage Keras session

def startSession(memory=0.3):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=memory)
    K.set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))

def finishSession():
    K.clear_session()
    K.get_session().close()