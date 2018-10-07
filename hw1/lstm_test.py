import tensorflow as tf
from keras.layers import Dropout, Dense, LSTM
from keras import backend as K
from keras.objectives import categorical_crossentropy

input = tf.placeholder(tf.float32, shape=(None, 784))
labels = tf.placeholder(tf.float32, shape=(None, 10))


x = Dense(128, activation='relu')(input)
# x = Dropout(0.5)(x)
# x = Dense(128, activation='relu')(x)
# x = Dropout(0.5)(x)
x = LSTM(32)(x)
preds = Dense(10)(x)

loss = tf.reduce_mean(categorical_crossentropy(labels, preds))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
# with sess.as_default():
    # for i in range(100):
        # batch = mnist_data.train.next_batch(50)
        # train_step.run(feed_dict={img: batch[0],
                                  # labels: batch[1],
                                  # K.learning_phase(): 1})

# acc_value = accuracy(labels, preds)
# with sess.as_default():
    # print acc_value.eval(feed_dict={img: mnist_data.test.images,
                                    # labels: mnist_data.test.labels,
                                    # K.learning_phase(): 0})
