from Environments.mindstormBot import mindstormBotEnv
import numpy as np
import torch
from Reward.rewardfuncs import sparse_reward2d
from math import pi
from Save_Gif.save_gif import save_frames_as_gif
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO, SAC, A2C, TD3
from gymnasium.vector import SyncVectorEnv
import gymnasium as gym
from gymnasium.envs.registration import register
from stable_baselines3.common.env_util import make_vec_env
from gymnasium import make
from stable_baselines3.common.vec_env import VecVideoRecorder
"""
todo:
add range finder
improve wall create obstacle
fix rotation bug at beginning

"""

import random
import os
device = torch.device('cuda')
render = True
register(
            id='mindstormBot-v0',  # Use a valid format, e.g., '<name>-v<version>'
            entry_point='Environments.mindstormBot:mindstormBotEnv',  # Update with your actual module and class
        )


#def make_env():
#    return gym.make("mindstormBot-v0")  # Incorrect, extra quote

if __name__ == '__main__':

    environment = 'mindstormBot'
    algorithm = 'PPO'
    training_timesteps = 10000
    t_s = 1/50
    #env = SyncVectorEnv([make_env for _ in range(num_envs)])
    env = make_vec_env(environment, n_envs=4)
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

    #for iteration in range(1, 1+1):
    #    model.learn(training_timesteps/10, reset_num_timesteps=False)
    """
        if iteration % 1 == 0:
            try:
                print(f"Rendering episode at iteration {iteration}") 
                for i in range(600):
                
                    action, _states = model.predict(obs, deterministic=True)
                    #if i % 10 == 0:
                    #    print("action and number of timesteps: ", i, action)
                    obs, reward, done, info = env.step(action)
                    #if i % 10 == 0:
                    #    print("observation: ",obs)

                    env.render(mode='rgb_array')
                    if done[0]:
                        obs = env.reset()
                env.close()
            except Exception as e:
                print(f"An error occurred: {e}. Skipping to the next iteration.")
                break
        
        if iteration % 3 == 0:
            try:
                run_name = "mindStormDec9_1222"+str(iteration)+"_"+str(training_timesteps)
                torch.save(model.policy.state_dict(), run_name + 'render.pt')
            except Exception as e:
                print(f"An error occurred: {e}. Skipping to the next iteration.")
                break
        """
    obs = env.reset()

    video_folder = "/home/ernst/workspaces/mindstorm_project_RL"
    video_length = 100
    env = VecVideoRecorder(env, video_folder,
                           record_video_trigger=lambda x: x == 0, video_length=video_length,
                           name_prefix=f"random-agent")
    env.reset()
    model = PPO('MlpPolicy', env, verbose=1, gamma=0.99, seed=seed)
    #model2.policy.load_state_dict(model)
    model.learn(total_timesteps=25000)
    for _ in range(video_length + 1):
      action = [env.action_space.sample(),env.action_space.sample(),env.action_space.sample(),env.action_space.sample()]

      obs, _, _, _ = env.step(action)
    # Save the video
    env.close()
    """
    frames = []
    # save gif at end of training
    single_env = make("mindstormBot-v0", render_mode='rgb_array')

    obs, info = single_env.reset()
    frames = []

    for i in range(2400):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, done2, info = single_env.step(action)
        print(obs)
        frames.append(env.render(mode='rgb_array'))
        if done:
            obs = env.reset()
    # Save Gif

    save_frames_as_gif(frames, filename=run_name+'.gif')
    """
    run_name = "dec11"+"_"+str(training_timesteps)

    # Save the final trained model
    torch.save(model.policy.state_dict(), run_name + '.pt')

    # Close the environment
    env.close()
