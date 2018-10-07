#!/usr/bin/env python

"""
Code to train a policy with dagger.
Example usage:
     python run_dagger.py expert_data/RoboschoolHalfCheetah-v1_60.pkl NameDagger
           --epochs 3000 --num_rollouts 100 --aggr_interval 100 --batch_size 128
"""

import os
import pickle
import tensorflow as tf
import numpy as np
import tf_util
import gym
from load_mypolicy import MyPolicy
import roboschool
import re
from load_zoopolicy import ZooPolicyTensorflow, SmallReactivePolicy


# dimensions used from the 
model_dims = {'RoboschoolHopper':       (15, 128, 64, 3),
              'RoboschoolAnt':          (28, 128, 64, 8),
              'RoboschoolHalfCheetah':  (26, 128, 64, 6),
              'RoboschoolHumanoid':     (44, 256, 128, 17),
              'RoboschoolWalker2d':     (22, 128, 64, 6),
              'RoboschoolReacher':      (9, 64, 32, 2),
             }

# _rollouts = (1, 5, 10, 20, 60, 100)
# _batch_sizes = (8, 16, 32, 64)
# _epochs = (10, 100, 1000, 10000, 20000)


def create_model(filename):
    dims = model_dims[re.compile('/|-').split(filename)[1]] # HERE ADDED BY NOVIN
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
    

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_data', type=str)
    parser.add_argument('trial_name', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, help='number of epochs for the training (both pre- and dagger- trainnig)')
    parser.add_argument("--test", action='store_true', help='to skip the pretraining rightaway to dagger')
    parser.add_argument("--num_rollouts", type=int, default=1, help='for the final evaluation')
    # dagger training once every 'rain_interval'
    parser.add_argument("--aggr_interval", type=int, default=10, help='how often update the policy model')
    # train model before the dagger 
    # parser.add_argument("--pretrain", action='store_true')

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

    _X = expert_data["observations"] # unnormalized observation
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

    print(_X.shape)
    print(y.shape)
    nr_observations = len(_X)
    assert len(_X) == len(y)

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

    print("pretraining..")
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

    if not args.test: # training
        train_policy(nr_epochs, batch_size, _X, y, sess, opt, mse)

    saver.restore(sess, 'models/{0}.ckpt'.format(trial_name_save))

    # dagger  ##################################################################
    print("start dagger..")
    if not args.test: # training
        with tf.Session():
            tf_util.initialize()

            import gym
            env = gym.make(envname)
            max_steps = args.max_timesteps or env.spec.timestep_limit
            if isReacher:
                pi = SmallReactivePolicy(env.observation_space, env.action_space)
            else:
                pi = ZooPolicyTensorflow("zoo_experts/"+envname, "mymodel1", 
                                         env.observation_space, env.action_space)

            returns = []
            observations = []
            expert_actions = []
            aggr_steps = 1 
            for i in range(args.num_rollouts):
                print('iter', i)
                obs = env.reset()
                done = False
                totalr = 0.
                steps = 0
                while not done:
                    if (aggr_steps % args.aggr_interval == 0): 
                        # update the policy with new observations (train the model)
                        _X = np.concatenate((_X, np.array(observations)))
                        y = np.concatenate((y, np.array(expert_actions)))
                        print("%i/%i > update policy"%(steps, max_steps))
                        print("\t    aggregated data: %i"%_X.shape[0])
                        observations = []
                        expert_actions = []

                        # initialize variables
                        sess.run(tf.global_variables_initializer())
                        train_policy(nr_epochs, batch_size, X, y, sess, opt, mse)

                        # with open(os.path.join("results", args.trial_name + "_" + envname), "a+") as f:
                            # f.write("%i, %f, %f\n" % (args.epochs, np.mean(returns_aggr), np.std(returns_aggr)))

                    action = sess.run(output_pred, feed_dict={input_ph: obs[np.newaxis,:]})
                    action = action.reshape(env.action_space.shape)
                    observations.append(obs)
                    expert_actions.append(pi.act(obs) if isReacher else pi.act(obs, env) 
)
                    obs, r, done, _ = env.step(action)
                    totalr += r
                    steps += 1
                    aggr_steps += 1
                    if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
                    if steps >= max_steps:
                        break

                returns.append(totalr)
                with open(os.path.join("results", args.trial_name + "_" + envname), "a+") as f:
                    f.write("%i, %f, %f\n" % (args.epochs, np.mean(returns), np.std(returns)))


    saver.restore(sess, 'models/{0}.ckpt'.format(trial_name_save))

    # run the environment for final evaluation  #################################
    print("evaluate dagger..")
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
                # observations.append(obs)
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


if __name__ == '__main__':
    main()
