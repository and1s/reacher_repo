
import gym
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pickle


from stable_baselines3 import PPO


# get birds-eye from env
def render_birds_eye(my_env):
    viewer = my_env.unwrapped.viewer
    camera_id = 0
    camera_distance = 0.8
    camera_azimuth = 0
    camera_elevation = -90
    
    if viewer is not None:
        viewer.cam.lookat[0] = 0
        viewer.cam.lookat[1] = 0
        viewer.cam.lookat[2] = 0
        viewer.cam.distance = camera_distance
        viewer.cam.azimuth = camera_azimuth
        viewer.cam.elevation = camera_elevation

    return my_env.render()


# run simulations to build dataset
def run_simulations(model, env, num_simulations = 100):
    data_pairs = []
    for i in range(num_simulations):
        if i % 10 == 0:
            print(f'Running simulation {i}...')
        obs = env.reset()[0]
        for t in range(50):
            image = render_birds_eye(env)
            down_scaled_image  = cv2.resize(image, (64, 64))
            data_pairs.append((down_scaled_image, obs))
            action = model.predict(obs)[0]
            obs, reward, terminated, truncated, info= env.step(action)
            if terminated or truncated:
                break
    return data_pairs


# load expert model for reacher
print('Loading expert Reacher agent...')
model = PPO.load("ppo_reacher_expert")


# make envs for model to run on
print('Making evaluation environment')
# env = gym.make("Reacher-v2", render_mode = 'rgb_array')
env = gym.make("Reacher-v4", render_mode = 'rgb_array')

env.reset()
render_birds_eye(env)

# build dataset
print('Building dataset...')
data_pairs = run_simulations(model, env)

env.close()
# save dataset in pkl file and overwrite
print('Saving dataset...')
with open('reacher_dataset.pkl', 'wb') as f:
    pickle.dump(data_pairs, f)

print('Dataset built!')

