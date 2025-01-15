from Environments.mindstormBot import mindstormBotEnv
import torch
from gymnasium.envs.registration import register
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecVideoRecorder, SubprocVecEnv
from sbx import PPO
from stable_baselines3 import DQN
import os
import shutil
from sb3_contrib import RecurrentPPO
import numpy as np

device = torch.device('cuda')
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
    training_timesteps = 1000000
    
    n_envs = 16 
    env = make_vec_env(environment, n_envs=n_envs, vec_env_cls=SubprocVecEnv)
    
    #create log dir
    log_dir = "/Desktop/workspaces/mindstorm_project_RL/logs/"
    algorithm_folder = "PPO_0"
    full_log_dir = os.path.join(log_dir, algorithm_folder)
    if os.path.isdir(full_log_dir):
        shutil.rmtree(full_log_dir)
        print(f"Deleted folder: {full_log_dir}")
    else:
        print(f"Folder not found: {full_log_dir}")

    #initialize and train model
    #model = PPO('MlpPolicy', env, verbose=1, gamma=0.99, clip_range=0.3 ,seed=None, tensorboard_log="/Desktop/workspaces/mindstorm_project_RL/logs/")
    #model = DQN('MlpPolicy', env, verbose=1, gamma=0.99,seed=None, batch_size=128,exploration_fraction=0.4, tensorboard_log="/Desktop/workspaces/mindstorm_project_RL/logs/")
    policy_kwargs = dict(
        lstm_hidden_size=64           # Use a small hidden size, e.g., 1 or 2
    )
    model = RecurrentPPO("MlpLstmPolicy", env,n_steps=64,policy_kwargs=policy_kwargs, verbose=1,tensorboard_log="/Desktop/workspaces/mindstorm_project_RL/logs/")

    obs = env.reset()
    model.learn(training_timesteps, reset_num_timesteps=False)
    env.close()
    lstm_states = None
    episode_starts = np.ones((n_envs,), dtype=bool)

    #create video
    video_folder = "/Desktop/workspaces/mindstorm_project_RL/videos/"
    video_length = 600
    env = make_vec_env(eval_environment, n_envs=16, vec_env_cls=SubprocVecEnv)
    env = VecVideoRecorder(env, video_folder,
                           record_video_trigger=lambda x: x == 0, video_length=video_length,
                           name_prefix=f"jan15_1406")
    env.reset()
    for _ in range(video_length):
        action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
        obs, rewards, dones, info = env.step(action)
        episode_starts = dones

    env.close()


    # Save the final trained model
    run_name = "jan15_1406"+"_"+str(training_timesteps)
    save_dir = "/Desktop/workspaces/mindstorm_project_RL/saved_models/"
    full_log_dir = os.path.join(save_dir, run_name)
    if not os.path.isdir(full_log_dir):
        os.makedirs(full_log_dir)
    model.save(full_log_dir)