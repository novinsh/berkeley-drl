#!/usr/bin/env python

"""
Code to load an expert policy and generate roll-out data for behavioral cloning.
Example usage:
    python run_expert.py experts/Humanoid-v1.pkl Humanoid-v1 --render \
            --num_rollouts 20

Author of this script and included expert policies: Jonathan Ho (hoj@openai.com)
"""

import os
import pickle
import tensorflow as tf
import numpy as np
import tf_util
import gym
# import load_policy
from load_zoopolicy import ZooPolicyTensorflow, SmallReactivePolicy
import roboschool
import re


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_policy_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    args = parser.parse_args()

    with tf.Session():
        tf_util.initialize()

        import gym
        env = gym.make(args.envname)

        print('loading and building expert policy')
        # policy_fn = load_policy.load_policy(args.expert_policy_file)
        isReacher = (re.compile('/|-').split(args.expert_policy_file)[1] == 'RoboschoolReacher')
        if isReacher:
            pi = SmallReactivePolicy(env.observation_space, env.action_space)
        else:
            pi = ZooPolicyTensorflow(args.expert_policy_file, "mymodel1", 
                                     env.observation_space, env.action_space)
        print('loaded and built')

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
                action = pi.act(obs) if isReacher else pi.act(obs, env) 
                observations.append(obs)
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

        print('returns', returns)
        print('mean return', np.mean(returns))
        print('std of return', np.std(returns))

        expert_data = {'observations': np.array(observations),
                       'actions': np.array(actions)}

        with open(os.path.join('expert_data', args.envname + '_' \
                    + str(args.num_rollouts) + '.pkl'), 'wb') as f:
            pickle.dump(expert_data, f, pickle.HIGHEST_PROTOCOL)

        with open(os.path.join('expert_data', args.envname + "_" \
                    + str(args.num_rollouts) + '_logs.txt'), 'w') as f:
            f.write("returns: {0}\n".format(returns))
            f.write("mean return: {0}\n".format(np.mean(returns)))
            f.write("std of return: {0}\n".format(np.std(returns)))


if __name__ == '__main__':
    main()
