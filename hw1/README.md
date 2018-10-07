# CS294-112 HW 1: Imitation Learning

Dependencies:
 * Python **3.5**
 * Numpy version **1.14.5**
 * TensorFlow version **1.10.5**
 * Roboschool version **1.1**
 * OpenAI Gym version **0.10.5**

Once Python **3.5** is installed, you can install the remaining dependencies using `pip install -r requirements.txt`.

In `experts/`, the provided expert policies are:
* RoboschoolAnt-v2.pkl
* RoboschoolHalfCheetah-v2.pkl
* RoboschoolHopper-v2.pkl
* RoboschoolHumanoid-v2.pkl
* RoboschoolReacher-v2.pkl
* RoboschoolWalker2d-v2.pkl
The name of the pickle file corresponds to the name of the gym environment.

## Directories

- expert_data: the pickled observation-actions pairs from each environment plus 
some logs storing the mean and std of reward out of each (the name of the files 
reflect the configuration of each log)
- models: the trained models as the result of immitation learning or dagger stored
hear for later execution or continuing the training
- results: some outputs that will be used directly in the answer of the tasks. e.g.
the hyperparameter task results stored here. The relevant jupyter-notebooks are also
stored here.
- zoo_expert: the weights for the Roboschool expert policies stored here! 

## Generate samples
you can execute ```./run_zooexpert``` to generate observations for the immitatio
learning step. It's similar to ./run_expert with the difference that it uses **Roboschool**
environment instead. You should make sure to create the directory expert_data before.

## Behavior Cloning

Example to run immitation without rendering to train:
```python run_bc.py expert_data/RoboschoolHalfCheetah-v1_60.pkl bc --epochs 10 --num_rollouts 60```

After training, the trained policy is put for evaluation over the same number of
specified rollouts. 

Example to run already cloned behavior with rendering to see final result:
```python run_bc.py expert_data/RoboschoolAnt-v1_60.pkl bc --epochs 10000 --num_rollouts 60 --test --render```


## Dagger

Example to run dagger:
```python run_daggernew.py expert_data/RoboschoolHumanoid-v1_60.pkl dagger --epochs 10 --num_rollouts 25 --dagger_iterations 40```

Example to run the trained dagger model with rendering: 
```python run_daggernew.py expert_data/RoboschoolHumanoid-v1_60.pkl dagger --epochs 10 --num_rollouts 4 --dagger_iterations 40 --test --render```


