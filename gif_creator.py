from Environments.crazyflie import Crazyflie_2d_inclined
import numpy as np
import torch
from Reward.rewardfuncs import sparse_reward2d, euclidean_reward3d
from math import pi
from Save_Gif.save_gif import save_frames_as_gif
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO, SAC, A2C, TD3

import random
import os
device = torch.device('cuda')
render = True

if __name__ == '__main__':
    device = torch.device('cuda')

    environment = 'CF_2d_inclined'  
    algorithm = 'PPO'
    training_timesteps = 3000000   
    t_s = 1/50                    

    if environment == 'CF_2d_inclined': 
        env = Crazyflie_2d_inclined(t_s, rewardfunc=sparse_reward2d)

    # Check if the environment is working right
    check_env(env)

    # Set seeds to be able to reproduce results
    seed = 1
    os.environ['PYTHONHASHSEED'] = str(seed)
    env.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Select the algorithm from Stable Baselines 3
    if algorithm == 'PPO':
        model = PPO('MlpPolicy', env, verbose=1, gamma=0.99, seed=seed) # Try 0.995 for inclined landing
    elif algorithm == 'SAC':
        model = SAC('MlpPolicy', env, verbose=1, gamma=0.97, seed=seed)
    elif algorithm == 'A2C':
        model = A2C('MlpPolicy', env, verbose=1, seed=seed)
    elif algorithm == 'TD3':
        model = TD3('MlpPolicy', env, verbose=1, gamma=0.97, seed=seed)
    
    #load pre-trained model
    #model = PPO('MlpPolicy', env, verbose=1, gamma=0.99, seed=seed)
    weights_path = '/home/ernst/thesis/InclinedDroneLander/may27B10_3000000render.pt'
    model.policy.load_state_dict(torch.load(weights_path, map_location='cuda'))
    model.policy.to('cuda')
    obs = env.reset()  # Reset the environment and get the initial observation

    frames = []
    # save gif at end of training
    for i in range(999):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        frames.append(env.render(mode='rgb_array'))
        if done:
            obs = env.reset()
    # Save Gif
    #iteration = 4500000
    #run_name = "may9_234445_3000000render"+str(iteration)+"_"+str(training_timesteps)
    run_name = "may27B_2render"
    save_frames_as_gif(frames, filename=run_name+'.gif')

