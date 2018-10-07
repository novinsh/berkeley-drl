#!/usr/bin/env python

"""
Code to train a policy with dagger.
Example usage:
     python run_dagger.py expert_data/RoboschoolHalfCheetah-v1_60.pkl NameDagger
           --epochs 3000 --num_rollouts 100 --aggr_interval 100 --batch_size 128
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
from load_zoopolicy import ZooPolicyTensorflow, SmallReactivePolicy


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_data', type=str)
    parser.add_argument('trial_name', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, help='number of epochs for the training')
    parser.add_argument("--test", action='store_true', help='to skip the pretraining rightaway to dagger')
    parser.add_argument("--num_rollouts", type=int, default=1, help='for the final evaluation')
    parser.add_argument("--dagger_iterations", type=int, default=10)

    args = parser.parse_args()
    nr_epochs = args.epochs
    batch_size = args.batch_size

    envname = re.compile('/|_').split(args.expert_data)[2]
    trial_name_save = args.trial_name + "_" + envname + "_" + \
                      str(nr_epochs) + "_" + str(batch_size)

    # expert_data = pickle.loads(args.expert_data)
    expert_data = None
    with open(args.expert_data, "rb") as f:
        expert_data = pickle.load(f)

    X = expert_data["observations"] # unnormalized observation
    y = expert_data["actions"]

    isReacher = (re.compile('/|-').split(args.expert_data)[1] == 'RoboschoolReacher')
    # expert_policy_file = os.path.join("zoo_experts", (envname.split('-')[0] + "_v1_2017jul.weights"))
    # print(expert_policy_file)

    # create tf session
    def tf_reset():
        try:
            sess.close()
        except:
            pass
        tf.reset_default_graph()
        return tf.Session()
    sess = tf_reset()

    print(X.shape)
    print(y.shape)
    nr_observations = len(X)
    assert len(X) == len(y)

    # print("all: ", np.std(X))
    # print("indi: ", np.std(X[1000,:]))
    # print(np.mean(X[10000,:]))
    # print(np.mean(X))
    # print(np.min(X))
    # print(np.max(X))
    # assert False

    # create model
    input_ph, output_ph, output_pred = create_model(args.expert_data)

    # create loss
    mse = tf.reduce_mean(0.5 * tf.square(output_pred - output_ph))

    # create optimizer
    opt = tf.train.AdamOptimizer().minimize(mse)

    # initialize variables
    sess.run(tf.global_variables_initializer())

    # create saver to save model variables
    saver = tf.train.Saver()

    if tf.train.checkpoint_exists('models/{0}.ckpt'.format(trial_name_save)):
        saver.restore(sess, 'models/{0}.ckpt'.format(trial_name_save))

    def train_policy(nr_epochs, batch_size, X, y, sess, opt, mse):
        for epoch in range(nr_epochs):
            for training_step in range(nr_observations//batch_size):
                indices = np.random.randint(low=0, high=nr_observations, size=batch_size)
                input_batch = X[indices, :]
                output_batch = y[indices, :]

                _, mse_run = sess.run([opt, mse],
                                      feed_dict={input_ph: input_batch, output_ph: output_batch})

            print('Epoch {0:0d}/{1:0d} mse: {2:.6f}'.format(epoch, nr_epochs, mse_run))
            saver.save(sess, 'models/{0}.ckpt'.format(trial_name_save))


    if args.test: # run a already trained dagger
        with tf.Session():
            tf_util.initialize()

            env = gym.make(envname)
            max_steps = args.max_timesteps or env.spec.timestep_limit

            returns = []
            actions = []
            for i in range(args.num_rollouts):
                print('iter', i)
                obs = env.reset()
                done = False
                totalr = 0.
                steps = 0
                while not done:
                    # action = env.action_space.sample()
                    action = sess.run(output_pred, feed_dict={input_ph: obs[np.newaxis,:]})
                    action = action.reshape(env.action_space.shape)
                    actions.append(action) # for distribution of the actions
                    obs, r, done, _ = env.step(action)
                    totalr += r
                    steps += 1
                    if args.render:
                        env.render()
                    if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
                    if steps >= max_steps:
                        break
                returns.append(totalr)

            # create a logs here!
            print('mean return', np.mean(returns))
            print('std of return', np.std(returns))
        return 

    ############################################################################
    # dagger    ###############################################################
    ############################################################################
    print("evaluate dagger..")
    with tf.Session():
        tf_util.initialize()

        env = gym.make(envname)
        max_steps = args.max_timesteps or env.spec.timestep_limit
        if isReacher:
            pi = SmallReactivePolicy(env.observation_space, env.action_space)
        else:
            pi = ZooPolicyTensorflow("zoo_experts/"+envname, "mymodel1", 
                                     env.observation_space, env.action_space)


        for dagger_step in range(args.dagger_iterations):
            print('Dagger step {0}'.format(dagger_step))
            train_policy(nr_epochs, batch_size, X, y, sess, opt, mse)

            returns = []
            observations = [] # D_pi
            actions = []
            expert_actions = []
            for i in range(args.num_rollouts):
                print('iter', i)
                obs = env.reset()
                done = False
                totalr = 0.
                steps = 0
                while not done:
                    # action = env.action_space.sample()
                    action = sess.run(output_pred, feed_dict={input_ph: obs[np.newaxis,:]})
                    action = action.reshape(env.action_space.shape)
                    observations.append(obs)
                    actions.append(action)
                    expert_actions.append(pi.act(obs) if isReacher else pi.act(obs, env))
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
            with open(os.path.join("results", args.trial_name + "_" + envname), "a+") as f:
                f.write("%i, %f, %f\n" % (dagger_step, np.mean(returns), np.std(returns)))

            # D <- D U D_pi
            X = np.concatenate((X, np.array(observations)))
            y = np.concatenate((y, np.array(expert_actions)))
            print("aggregated data: %i"%X.shape[0])


        train_policy(nr_epochs, batch_size, X, y, sess, opt, mse)


if __name__ == '__main__':
    main()
