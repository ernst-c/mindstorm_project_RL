from Environments.mindstormBot import mindstormBotEnv
import torch
from gymnasium.envs.registration import register
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecVideoRecorder, SubprocVecEnv
from sb3_contrib import RecurrentPPO
import os
import numpy as np

# Set the device for PyTorch
device = torch.device('cuda')

# Register the environments
register(
    id='mindstormBot-v0',
    entry_point='Environments.mindstormBot:mindstormBotEnv',
)
register(
            id='mindstormBotEval-v0',  # Use a valid format, e.g., '<name>-v<version>'
            entry_point='Eval_Environments.mindstormBot:mindstormBotEnv',  # Update with your actual module and class
        )


if __name__ == '__main__':
    # Define variables
    eval_environment = 'mindstormBotEval-v0'

    training_timesteps = 800000  # Number of timesteps for additional training
    saved_model_dir = "/Desktop/workspaces/mindstorm_project_RL/saved_models/jan16_working_1459_1_25million.zip"

    n_envs = 16
    env = make_vec_env(eval_environment, n_envs=n_envs, vec_env_cls=SubprocVecEnv)

    # Load the saved model
    model = RecurrentPPO.load(saved_model_dir, env=env, device=device)

    obs = env.reset()
    lstm_states = None
    episode_starts = np.ones((n_envs,), dtype=bool)

    #create video
    video_folder = "/Desktop/workspaces/mindstorm_project_RL/videos/"
    video_length = 600
    env = VecVideoRecorder(env, video_folder,
                           record_video_trigger=lambda x: x == 0, video_length=video_length,
                           name_prefix=f"renderfps_jan16_working_1459_1_25million")
    env.reset()
    for _ in range(video_length):
        action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
        obs, rewards, dones, info = env.step(action)
        episode_starts = dones

    env.close()
