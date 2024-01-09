
import gym
import numpy as np
import matplotlib.pyplot as plt
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


# TRAIN reacher
print('Making training environment')
vec_env = make_vec_env("Reacher-v2", n_envs=4)

print(torch.cuda.is_available())

print('Training Expert Reacher...')

# start from scratch
# model = PPO("MlpPolicy", vec_env, verbose=1, gamma = 0.9)

# start from previous expert
model = PPO.load('ppo_reacher_expert', env = vec_env)

model.learn(total_timesteps=100000)
model.save("ppo_reacher_expert")

# EVALUATE REACHER
#make vec_env for model to run on
print('Making evaluation environment')
vec_env = make_vec_env("Reacher-v2", n_envs=4)

#load expert model for reacher
model = PPO.load("ppo_reacher_expert")

#test capabilities
print('Evaluating expert Reacher...')
mean_reward, std_reward = evaluate_policy(model, vec_env, n_eval_episodes=100)
print(f"Mean reward: {mean_reward} +/- {std_reward}")