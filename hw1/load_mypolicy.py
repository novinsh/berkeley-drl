import tensorflow as tf
import re
import tensorflow as tf
from keras.layers import Dropout, Dense, LSTM
from keras import backend as K
from keras.objectives import categorical_crossentropy

# using different dimensions for each model
model_dims = {'RoboschoolHopper':           (15, 128, 64, 3),
              'RoboschoolAnt':              (28, 128, 64, 8),
              'RoboschoolHalfCheetah':      (26, 128, 64, 6),
              'RoboschoolHumanoid':         (44, 256, 128, 17),
              'RoboschoolWalker2d':         (22, 128, 64, 6),
              'RoboschoolReacher':          (9, 64, 32, 2),
             }

def create_model(filename):
        dims = model_dims[re.compile('/|-').split(filename)[1]]

        # create inputs
        input_ph = tf.placeholder(dtype=tf.float32, shape=[None, dims[0]])
        output_ph = tf.placeholder(dtype=tf.float32, shape=[None, dims[3]])

        # create variables
        W0 = tf.get_variable(name='W0', shape=[dims[0], dims[1]], \
                             initializer=tf.contrib.layers.xavier_initializer())
        W1 = tf.get_variable(name='W1', shape=[dims[1], dims[2]], \
                             initializer=tf.contrib.layers.xavier_initializer())
        W2 = tf.get_variable(name='W2', shape=[dims[2], dims[3]], \
                             initializer=tf.contrib.layers.xavier_initializer())

        b0 = tf.get_variable(name='b0', shape=[dims[1]], initializer=tf.constant_initializer(0.))
        b1 = tf.get_variable(name='b1', shape=[dims[2]], initializer=tf.constant_initializer(0.))
        b2 = tf.get_variable(name='b2', shape=[dims[3]], initializer=tf.constant_initializer(0.))

        weights = [W0, W1, W2]
        biases = [b0, b1, b2]
        activations = [tf.nn.relu, tf.nn.relu, None]

        # create computation graph
        layer = input_ph
        for W, b, activation in zip(weights, biases, activations):
            layer = tf.matmul(layer, W) + b
            if activation is not None:
                layer = activation(layer)
        output_pred = layer
        
        return input_ph, output_ph, output_pred
        


def create_model_recurrent(filename, window_size):
    dims = model_dims[re.compile('/|-').split(filename)[1]]

    # input_ph = tf.placeholder(tf.float32, shape=(none, dims[0]))
    # output_ph = tf.placeholder(tf.float32, shape=(none, dims[3]))

    input_ph = tf.placeholder(tf.float32, shape=(None, window_size, dims[0]))
    output_ph = tf.placeholder(tf.float32, shape=(None, dims[3]))

    # x = Dense(dims[1], activation='relu')(input_ph)
    # x = Dropout(0.5)(x)
    # x = Dense(128, activation='relu')(x)
    x = LSTM(dims[2], return_sequences=True)(input_ph)
    x = Dropout(0.25)(x)
    x = LSTM(dims[2])(x)
    output_pred = Dense(dims[3])(x)

    return input_ph, output_ph, output_pred


