

import gym
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pickle


from stable_baselines3 import PPO

# run simulations to build dataset
def run_simulations(model, env, num_simulations = 1000):
    reacher_full_demo_data = []
    for i in range(num_simulations):
        traj_obs = []
        traj_actions = []
        traj_next_obs = []
        if i % 50 == 0:
            print(f'Running simulation {i}...')
        obs = env.reset()[0]
        for t in range(50):
            action = model.predict(obs)[0]
            next_obs, reward, terminated, truncated, info = env.step(action)
            traj_obs.append(obs)
            traj_actions.append(action)
            traj_next_obs.append(next_obs)
            if terminated or truncated:
                break
            obs = next_obs
        traj_obs = np.array(traj_obs)
        traj_actions = np.array(traj_actions)
        traj_next_obs = np.array(traj_next_obs)
        reacher_full_demo_data.append({'obs': traj_obs, 'action': traj_actions, 'next_obs': traj_next_obs})
    return reacher_full_demo_data


def create_id_ood_goals(env):
    full_id_goals = []
    full_ood_goals = []

    while len(full_id_goals) < 50 or len(full_ood_goals) < 50:
        obs = env.reset()[0]
        target_x = obs[4]
        target_y = obs[5]
        if target_y > 0 and len(full_id_goals) < 50:
            full_id_goals.append(np.array([target_x, target_y]))
        elif target_y < 0 and len(full_ood_goals) < 50:
            full_ood_goals.append(np.array([target_x, target_y]))

    full_id_goals = np.array(full_id_goals)
    full_ood_goals = np.array(full_ood_goals)
    return full_id_goals, full_ood_goals


# load expert model for reacher
print('Loading expert Reacher agent...')
model = PPO.load("ppo_reacher_expert")

# make envs for model to run on
print('Making evaluation environment')
env = gym.make("Reacher-v4")
env.reset()

# run simulations
print('Running simulations...')
reacher_full_demo_data = run_simulations(model, env, num_simulations = 1000)

# save data
print('Saving data...')
with open('demos.pkl', 'wb') as f:
    pickle.dump(reacher_full_demo_data, f)


# create id and ood goals
print('Creating ID and OOD goals...')
full_id_goals, full_ood_goals = create_id_ood_goals(env)

# save goals
print('Saving goals...')
with open('in_dist_goals.pkl', 'wb') as f:
    pickle.dump(full_id_goals, f)

with open('ood_goals.pkl', 'wb') as f:
    pickle.dump(full_ood_goals, f)
