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

    environment = 'CF_2d_inclined'  
    algorithm = 'PPO'
    training_timesteps = 30000
    t_s = 1/50                    

    if environment == 'CF_2d_inclined': 
        env = Crazyflie_2d_inclined(t_s, rewardfunc=sparse_reward2d)

    check_env(env)

    seed = 1
    os.environ['PYTHONHASHSEED'] = str(seed)
    env.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
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
    

    for iteration in range(1, 10 + 1):
        model.learn(training_timesteps/30, reset_num_timesteps=False)
        
        if iteration % 1 == 0:
            try:
                print(f"Rendering episode at iteration {iteration}") 
                for i in range(900):
                
                    action, _states = model.predict(obs, deterministic=True)
                    if i % 10 == 0:
                        print("action and number of timesteps: ", i, action)
                    obs, reward, done, info = env.step(action)  
                    env.render(mode='human')
                    if done:
                        obs = env.reset()
                env.close()
            except Exception as e:
                print(f"An error occurred: {e}. Skipping to the next iteration.")
                break
        
        if iteration % 5 == 0:    
            try:
                run_name = "may27B"+str(iteration)+"_"+str(training_timesteps)
                torch.save(model.policy.state_dict(), run_name + 'render.pt')
            except Exception as e:
                print(f"An error occurred: {e}. Skipping to the next iteration.")
                break
                
    frames = []
    # save gif at end of training
    for i in range(2400):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        frames.append(env.render(mode='rgb_array'))
        if done:
            obs = env.reset()
    # Save Gif
    run_name = "may27B"+str(iteration)+"_"+str(training_timesteps)

    save_frames_as_gif(frames, filename=run_name+'.gif')

    # Save the final trained model
    torch.save(model.policy.state_dict(), run_name + '.pt')

    # Close the environment
    env.close()
