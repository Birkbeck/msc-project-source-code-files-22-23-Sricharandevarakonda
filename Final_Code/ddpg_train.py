import numpy as np
import cv2

from car_environment import CarEnv
from ddpg_torch import AgentCNN

VIDEO_SAVE_FREQUENCY = 1000
CHECKPOINT_SAVE_FREQUENCY = 10000
N_EPISODES = 50000

env = CarEnv(model="DDPG", traffic=False, random_weather=False)

agent = AgentCNN(alpha=0.000025, beta=0.00025, input_dims=[3, 200, 300], tau=0.001, env=env,
              batch_size=64,  layer1_size=400, layer2_size=300, n_actions=2)
np.random.seed(0)
model_name="dqn_without_traffic"

score_history = []

# Main training loop
for i in range(N_EPISODES):
    try:
        _ = env.reset(i)
    except Exception as e:
        print("Exception occured :", e)
    if i % VIDEO_SAVE_FREQUENCY == 0:
        env.set_CAMERA_ON("Training_Videos", model_name, i)

    cv_frame = cv2.resize(env.render(), dsize=(200,300), interpolation = cv2.INTER_AREA)
    old_frame = np.reshape(cv2.cvtColor(cv_frame, cv2.COLOR_RGB2BGR), (1, 3, 200, 300))      
    done = False
    score = 0
    while not done:
        # Choose action
        act = agent.choose_action(old_frame)[0]

        # Execute action
        new_state, reward, done, info = env.step(act)


        cv_frame = cv2.resize(env.render(), dsize=(200,300), interpolation = cv2.INTER_AREA)
        new_frame = np.reshape(cv2.cvtColor(cv_frame, cv2.COLOR_RGB2BGR), (1, 3, 200, 300))

        agent.remember(old_frame, act, reward, new_frame, int(done))
        old_frame = new_frame.copy()
        
        agent.learn()
        score += reward
        obs = new_state
    if i % CHECKPOINT_SAVE_FREQUENCY == 0:
        agent.save_models(i*1000)
    score_history.append(score)


    print('Episode ', i, 'score %.2f' % score)