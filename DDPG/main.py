from ddpg_torch import AgentCNN
import gymnasium as gym
import numpy as np
from utils import plotLearning
import cv2
import os
from car_environment import CarEnv

#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:2048"

env = gym.make('LunarLanderContinuous-v2', render_mode="rgb_array")
env = CarEnv()
agent = AgentCNN(alpha=0.000025, beta=0.00025, input_dims=[3, 200, 300], tau=0.001, env=env,
              batch_size=64,  layer1_size=400, layer2_size=300, n_actions=2)

#agent.load_models()
np.random.seed(0)

score_history = []
for i in range(1000):
    try:
        _ = env.reset()
    except Exception as e:
        continue
    if i % 10 == 0:
        env.set_show_cam()
    _frame = cv2.resize(env.render(), dsize=(200,300), interpolation = cv2.INTER_AREA)
    old_frame = np.reshape(cv2.cvtColor(_frame, cv2.COLOR_RGB2BGR), (1, 3, 200, 300))  
    # resize image
    done = False
    score = 0
    while not done:
        act = agent.choose_action(old_frame)[0]
        new_state, reward, done, info = env.step(act)
        _frame = cv2.resize(env.render(), dsize=(200,300), interpolation = cv2.INTER_AREA)
        new_frame = np.reshape(cv2.cvtColor(_frame, cv2.COLOR_RGB2BGR), (1, 3, 200, 300))
        # resize image
        agent.remember(old_frame, act, reward, new_frame, int(done))
        old_frame = new_frame.copy()
        agent.learn()
        score += reward
        obs = new_state
    if i % 500 == 0:
        agent.save_models()
    score_history.append(score)

    #if i % 25 == 0:
    #    agent.save_models()

    print('episode ', i, 'score %.2f' % score,
          'trailing 100 games avg %.3f' % np.mean(score_history[-100:]))

filename = 'LunarLander-alpha000025-beta00025-400-300.png'
plotLearning(score_history, filename, window=100)
