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

if __name__ == '__main__':

    environment = 'CF_2d_inclined'
    algorithm = 'PPO'               
    training_timesteps = 20000001   
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

    # Select the algorithm from Stable Baselines 3
    if algorithm == 'PPO':
        model = PPO('MlpPolicy', env, verbose=1, gamma=0.97, seed=seed)
    elif algorithm == 'SAC':
        model = SAC('MlpPolicy', env, verbose=1, gamma=0.97, seed=seed)
    elif algorithm == 'A2C':
        model = A2C('MlpPolicy', env, verbose=1, seed=seed)
    elif algorithm == 'TD3':
        model = TD3('MlpPolicy', env, verbose=1, gamma=0.97, seed=seed)

    run_name = "may27B10"

    weights_path = '/home/ernst/thesis/InclinedDroneLander/may27B10_3000000render.pt'
    model.policy.load_state_dict(torch.load(weights_path, map_location=device))
    model.policy.to(device)

    model.policy.eval()

    obs = env.reset()
    example_input = torch.Tensor([obs]).to(device)

    traced_script_module = torch.jit.trace(model.policy, example_input, check_trace=False).to(device)
    traced_script_module.save(run_name + 'traced_model.pt')

    frames = []

    env.close()







