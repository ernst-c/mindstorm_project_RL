from Environments.mindstormBot import mindstormBotEnv
import numpy as np
import torch
from Reward.rewardfuncs import sparse_reward2d
from math import pi
from Save_Gif.save_gif import save_frames_as_gif
from stable_baselines3.common.env_checker import check_env
#from stable_baselines3 import PPO, SAC, A2C, TD3
from gymnasium.vector import SyncVectorEnv
import gymnasium as gym
from gymnasium.envs.registration import register
from stable_baselines3.common.env_util import make_vec_env
from gymnasium import make
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder, SubprocVecEnv
from sbx import DDPG, DQN, PPO
"""
todo:
look at seeds
change dynamic model? --> seems to work now, that timestep has been decreased
only range finder
more complicated track
"""
from stable_baselines3.common.callbacks import EvalCallback

import random
import os
import shutil
device = torch.device('cuda')
render = True
register(
            id='mindstormBot-v0',  # Use a valid format, e.g., '<name>-v<version>'
            entry_point='Environments.mindstormBot:mindstormBotEnv',  # Update with your actual module and class
        )
register(
            id='mindstormBotEval-v0',  # Use a valid format, e.g., '<name>-v<version>'
            entry_point='Eval_Environments.mindstormBot:mindstormBotEnv',  # Update with your actual module and class
        )

if __name__ == '__main__':

    environment = 'mindstormBot-v0'
    eval_environment = 'mindstormBotEval-v0'
    algorithm = 'PPO'
    training_timesteps = 6000000
    n_envs = 16 
    env = make_vec_env(environment, n_envs=n_envs, vec_env_cls=SubprocVecEnv)
    #eval_env = make_vec_env(eval_environment, n_envs=n_envs, vec_env_cls=SubprocVecEnv)

    #eval_callback = EvalCallback(eval_env,
    #                            log_path="./logs/", eval_freq=50000,
    #                            deterministic=True, render=True)

    #check_env(env)

    seed = 1
    os.environ['PYTHONHASHSEED'] = str(seed)
    #env.seed(seed)
    #random.seed(seed)
    #torch.manual_seed(seed)
    #np.random.seed(seed)
    log_dir = "/Desktop/workspaces/mindstorm_project_RL/logs/"
    algorithm_folder = "PPO_0"  # Folder name to delete
    full_log_dir = os.path.join(log_dir, algorithm_folder)
    shutil.rmtree(full_log_dir)  # Delete the folder and its contents

    if algorithm == 'PPO':
        model = PPO('MlpPolicy', env, verbose=1, gamma=0.99, seed=None, tensorboard_log="/Desktop/workspaces/mindstorm_project_RL/logs/")
    elif algorithm == 'SAC':
        model = SAC('MlpPolicy', env, verbose=1, gamma=0.97, seed=seed)
    elif algorithm == 'A2C':
        model = A2C('MlpPolicy', env, verbose=1, seed=seed)
    elif algorithm == 'TD3':
        model = TD3('MlpPolicy', env, verbose=1, gamma=0.97, seed=seed)

    #load pre-trained model
    #model = PPO('MlpPolicy', env, verbose=1, gamma=0.99, seed=seed)
    #weights_path = '/home/ernst/thesis/InclinedDroneLander/may11_A20_3000000render.pt'
    #model.policy.load_state_dict(torch.load(weights_path, map_location=device))
    #model.policy.to(device)
    obs = env.reset()
    #model.learn(training_timesteps, reset_num_timesteps=False,callback=eval_callback)
    model.learn(training_timesteps, reset_num_timesteps=False)
    env.close()

    video_folder = "/Desktop/workspaces/mindstorm_project_RL"
    video_length = 600

    environment = 'mindstormBotEval-v0'
    env = make_vec_env(environment, n_envs=16, vec_env_cls=SubprocVecEnv)
    env = VecVideoRecorder(env, video_folder,
                           record_video_trigger=lambda x: x == 0, video_length=video_length,
                           name_prefix=f"random-agent")
    env.reset()
    for _ in range(video_length):
        actions = model.predict(obs)[0] #[-0.5,0.5] #[x,y,0,0,theta,laser_range_finder_range] [laser_range_finder_range]
        obs, rewards, dones, info = env.step(actions)
        #for i in range(n_envs):
        #    if dones[i]:
        #        obs[i] = env.reset()[i]
    env.close()

    run_name = "dec18_1155"+"_"+str(training_timesteps)

    # Save the final trained model
    torch.save(model.policy.state_dict(), run_name + '.pt')

    # Close the environment
    env.close()
