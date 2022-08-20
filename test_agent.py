from __future__ import print_function

from datetime import datetime
import numpy as np
import gym
import os
import json
import torch
from model import pyTorchModel

def run_episode(env, agent, rendering=True, max_timesteps=500):
    
    episode_reward = 0
    step = 0

    state = env.reset()
    while True:
    
        # get action
        agent.eval()
        tensor_state = torch.from_numpy(state).float()
        tensor_action = agent(tensor_state)
        a = tensor_action.detach().numpy()[0]

        next_state, r, done, info = env.step(a)   
        episode_reward += r       
        state = next_state
        step += 1
        
        if rendering:
            env.render()

        if step > max_timesteps: 
            break

    return episode_reward


if __name__ == "__main__":

    # important: don't set rendering to False for evaluation (you may get corrupted state images from gym)
    rendering = False                      
    
    n_test_episodes = 10                # number of episodes to test

    # load agent
    agent = pyTorchModel(belief_dim=4, obs_dim=4, actor_hidden_dim=64, action_dim=1)
    agent.load_state_dict(torch.load("models/InvertedPendulum.pkl"))

    env = gym.make("InvertedPendulum-v2")

    episode_rewards = []
    for i in range(n_test_episodes):
        episode_reward = run_episode(env, agent, rendering=rendering)
        episode_rewards.append(episode_reward)

    # save results in a dictionary and write them into a .json file
    results = dict()
    results["episode_rewards"] = episode_rewards
    results["mean"] = np.array(episode_rewards).mean()
    results["std"] = np.array(episode_rewards).std()
 
    fname = "results/results_bc_agent-%s.json" % datetime.now().strftime("%Y%m%d-%H%M%S")
    fh = open(fname, "w")
    json.dump(results, fh)
            
    env.close()
    print('... finished')
