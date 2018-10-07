#!/usr/bin/env python

"""
Code to load an expert policy and perform behavior cloning.
Example usage:

"""

import os
import re
import pickle
import tensorflow as tf
import numpy as np
import gym
import tf_util
import roboschool
from load_mypolicy import *

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_data', type=str)
    parser.add_argument('trial_name', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--test", action='store_true')
    parser.add_argument("--num_rollouts", type=int, default=1)

    args = parser.parse_args()

    envname = re.compile('/|_').split(args.expert_data)[2]
    trial_name_save = args.trial_name + "_" + envname + "_" + \
                                      str(args.epochs) + "_" + str(args.batch_size)

    # load expert data 
    expert_data = None
    with open(args.expert_data, "rb") as f:
        expert_data = pickle.load(f)
    X = expert_data["observations"]
    y = expert_data["actions"]

    print(X.shape)
    print(y.shape)
    nr_observations = len(X)
    assert len(X) == len(y)

    # create tf session
    def tf_reset():
        try:
            sess.close()
        except:
            pass
        tf.reset_default_graph()
        return tf.Session()
    sess = tf_reset()


    # create model
    # input_ph, output_ph, output_pred = create_model(args.expert_data)
    window = 10
    input_ph, output_ph, output_pred = create_model_recurrent(args.expert_data, window)

    # create loss
    mse = tf.reduce_mean(0.5 * tf.square(output_pred - output_ph))

    # create optimizer
    opt = tf.train.AdamOptimizer().minimize(mse)

    # initialize variables
    sess.run(tf.global_variables_initializer())

    # create saver to save model variables
    saver = tf.train.Saver()

    def train_policy(epochs, batch_size, X, y, sess, opt, mse):
        for epoch in range(epochs):
            for training_step in range(nr_observations//batch_size):
                indices = np.random.randint(low=0, high=nr_observations, size=args.batch_size)
                input_batch = X[indices, :]
                output_batch = y[indices, :]

                _, mse_run = sess.run([opt, mse],
                                      feed_dict={input_ph: input_batch, output_ph: output_batch})

            print('Epoch {0:0d}/{1:0d} mse: {2:.6f}'.format(epoch, epochs, mse_run))
            saver.save(sess, 'models/{0}.ckpt'.format(trial_name_save))

    def train_policy_recurrent(epochs, batch_size, X, y, sess, opt, mse):
        for epoch in range(epochs):
            for training_step in range(nr_observations//batch_size-window):
                # indices = np.random.randint(low=0, high=nr_observations, size=args.batch_size)
                input_batch = None
                output_batch = None
                for w in range(window):
                    batch_beg = training_step * (batch_size+window-1) + w
                    batch_end = batch_beg + batch_size
                    if input_batch is None:
                        input_batch = X[batch_beg:batch_end, :][:, np.newaxis, :]
                    else:
                        input_batch = np.hstack((input_batch, X[batch_beg:batch_end, np.newaxis, :]))
                    output_batch = y[batch_beg:batch_end, :]
                print(input_batch.shape)
                print(output_batch.shape)

                print(output_batch.shape)
                _, mse_run = sess.run([opt, mse],
                                      feed_dict={input_ph: input_batch, output_ph: output_batch})
                print("fiished rain %d / %d "% (training_step, nr_observations//batch_size-window))

            print('Epoch {0:0d}/{1:0d} mse: {2:.6f}'.format(epoch, epochs, mse_run))
            saver.save(sess, 'models/{0}.ckpt'.format(trial_name_save))



    # load the model if already exists
    if tf.train.checkpoint_exists('models/{0}.ckpt'.format(trial_name_save)):
        saver.restore(sess, 'models/{0}.ckpt'.format(trial_name_save))

    if not args.test: # training
        train_policy_recurrent(args.epochs, args.batch_size, X, y, sess, opt, mse)

    # run the environment for evaluation
    with tf.Session():
        tf_util.initialize()

        import gym
        env = gym.make(envname)
        max_steps = args.max_timesteps or env.spec.timestep_limit

        returns = []
        observations = []
        actions = []
        for i in range(args.num_rollouts):
            print('iter', i)
            obs = env.reset()
            done = False
            totalr = 0.
            steps = 0
            while not done:
                if len(observations) < window:
                    action = env.action_space.sample()
                else:
                    observation_seq = None
                    for w in window:

                    action = sess.run(output_pred, feed_dict={input_ph: obs[np.newaxis,:]})
                action = action.reshape(env.action_space.shape)
                actions.append(action)
                obs, r, done, _ = env.step(action)
                totalr += r
                steps += 1
                if args.render:
                    env.render()
                if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
                if steps >= max_steps:
                    break
            returns.append(totalr)

        print('mean return', np.mean(returns))
        print('std of return', np.std(returns))

    # append the results to he file
    with open(os.path.join("results", args.trial_name + "_" + envname), "a+") as f:
        f.write("%i, %f, %f\n" % (args.epochs, np.mean(returns), np.std(returns)))


if __name__ == '__main__':
    main()
