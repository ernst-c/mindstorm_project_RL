from Environments.mindstormBot import mindstormBotEnv
import torch
from gymnasium.envs.registration import register
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from sb3_contrib import RecurrentPPO
import os

# Set the device for PyTorch
device = torch.device('cuda')

# Register the environments
register(
    id='mindstormBot-v0',
    entry_point='Environments.mindstormBot:mindstormBotEnv',
)

if __name__ == '__main__':
    # Define variables
    environment = 'mindstormBot-v0'
    training_timesteps = 250000  # Number of timesteps for additional training
    saved_model_dir = "/Desktop/workspaces/mindstorm_project_RL/saved_models/jan16_working_1459_1million.zip"

    n_envs = 16
    env = make_vec_env(environment, n_envs=n_envs, vec_env_cls=SubprocVecEnv)

    # Load the saved model
    model = RecurrentPPO.load(saved_model_dir, env=env, device=device)

    # Continue training
    model.learn(training_timesteps, reset_num_timesteps=False)
    env.close()

    # Save the updated model
    new_run_name = "jan16_working_1459_1_25million"
    new_save_dir = "/Desktop/workspaces/mindstorm_project_RL/saved_models/"
    new_full_log_dir = os.path.join(new_save_dir, new_run_name)
    if not os.path.isdir(new_full_log_dir):
        os.makedirs(new_full_log_dir)
    model.save(new_full_log_dir)
    print(f"Model saved to {new_full_log_dir}")
