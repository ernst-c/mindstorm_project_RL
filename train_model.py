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
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from sbx import DDPG, DQN, PPO
"""
todo:

"""

import random
import os
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

#def make_env():
#    return gym.make("mindstormBot-v0")  # Incorrect, extra quote

if __name__ == '__main__':

    environment = 'mindstormBot'
    algorithm = 'PPO'
    training_timesteps = 3000000
    t_s = 1/50
    #env = SyncVectorEnv([make_env for _ in range(num_envs)])
    n_envs = 8
    env = make_vec_env(environment, n_envs=n_envs)
    #if environment == 'mindstormBot':
    #    env = mindstormBot(t_s, rewardfunc=sparse_reward2d)

    #check_env(env)

    seed = 1
    os.environ['PYTHONHASHSEED'] = str(seed)
    #env.seed(seed)
    #random.seed(seed)
    #torch.manual_seed(seed)
    #np.random.seed(seed)

    if algorithm == 'PPO':
        model = PPO('MlpPolicy', env, verbose=1, gamma=0.99, seed=seed)
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
    model.learn(training_timesteps, reset_num_timesteps=False)

    video_folder = "/Desktop/workspaces/mindstorm_project_RL"
    video_length = 500
    
    environment = 'mindstormBotEval-v0'
    env = make_vec_env(environment, n_envs=8)
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

    run_name = "dec15_1628"+"_"+str(training_timesteps)

    # Save the final trained model
    torch.save(model.policy.state_dict(), run_name + '.pt')

    # Close the environment
    env.close()
