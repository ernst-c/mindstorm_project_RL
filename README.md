This repository contains a custom gymnasium environment which can be used to train a RL agent to navigate a robot through a maze.

branches:
- discrete_action_space --> contains basis code for training with a discrete action space
- curriculum_learning_disc_actions --> unfinished curriculum learning approach
- recurrent_ppo --> uses recurrent ppo policy, same as discrete_action_space
- checkpoints_rewardfunc_disc_actions --> reward function has positive reward for reaching certain checkpoints en route to goal. 
- checkpoints_recurrent_ppo --> uses recurrent ppo policy, same as checkpoints_rewardfunc_disc_actions

- lrf_obs_disc --> observation space only contains laser range finder information, branched from discrete action space
- lrf_obs_disc_checkpoints --> observation space only contains laser range finder information, branched from checkpoints_rewardfunc_disc_actions

- continuous_action_space --> deprecated branch, action space consists of left and right wheel rotational velocity
- google_collab --> deprecated
- normalized_observation_space --> deprecated

To enable rendering from docker container, type following in terminal: 
```bash
xhost +
```
To create Docker image:
```bash
cd mindstorm_project_rl
docker build -t gymnasium-env -f Dockerfile .
```
To run Docker image (change mounted volume):
```bash
docker run --rm -it --cpus=16 --memory=12G --name gymnasium-env -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v /c/Users/elcan/Desktop:/Desktop -p 6006:6006 gymnasium-env /bin/bash
```
To run tensorboard run the following from outside docker container terminal:
```bash
cd mindstorm_project_rl
tensorboard --logdir=logs
```
When running train.py, make sure n_envs does not exceed amount of logical processors on PC. 
