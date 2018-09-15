#!/bin/bash
set -eux
for e in RoboschoolHopper-v1 RoboschoolAnt-v1 RoboschoolHalfCheetah-v1 RoboschoolHumanoid-v1 RoboschoolWalker2d-v1 RoboschoolReacher-v1 
do
    python run_zooexpert.py zoo_experts/$e.weights $e --render --num_rollouts=10
done
