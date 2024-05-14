#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 필요 라이브러리 임포트
import gym
import argparse
from ns3gym import ns3env
import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow as tf
# import tensorflow.contrib.slim as slim
# import tf_slim as slim
from tensorflow import keras
import numpy as np
from codecarbon import EmissionsTracker
from comet_ml import Experiment

__author__ = "Lee Sungjin"
__copyright__ = "Copyright (c) 2024, Lee Sungjin"
__version__ = "0.1.0"
__email__ = "ssjj3552@gmail.com"

tracker = EmissionsTracker(
    project_name="centralized",
    measure_power_secs=10,
    save_to_file=True
)

parser = argparse.ArgumentParser(description='Start simulation script on/off')
parser.add_argument('--start',
					type=int,
					default=1,
					help='Start ns-3 simulation script 0/1, Default: 1')
parser.add_argument('--iterations',
					type=int,
					default=1,
					help='Number of iterations, Default: 1')
parser.add_argument('--steps',
					type=int,
					default=100,
					help='Number of steps, Default 100')
args = parser.parse_args()

startSim = bool(args.start)
iterationNum = int(args.iterations)
maxSteps = int(args.steps)

port = 5555
simTime = maxSteps / 10.0 # seconds
seed = 12

# create environment
env = ns3env.Ns3Env(port=port, startSim=startSim, simSeed=seed)
ob_space = env.observation_space
ac_space = env.action_space
print("Observation space: ", ob_space,  ob_space.dtype)
print("Action space: ", ac_space, ac_space.dtype)

def build_model(input_size, output_size):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(state_size, activation='relu', input_shape=(state_size,)),
        tf.keras.layers.Dense(state_size/2, activation='relu'),
        tf.keras.layers.Dense(action_size, activation='softmax')
    ])
    # model = tf.keras.Sequential([
    #     tf.keras.layers.Dense(128, activation='relu', input_shape=(state_size,)),
    #     tf.keras.layers.Dense(64, activation='relu'),
    #     tf.keras.layers.Dense(32, activation='relu'),
    #     tf.keras.layers.Dense(action_size, activation='softmax')
    # ])
    return model

state_size = ob_space.shape[0]
action_size = ac_space.n

# build model
model = build_model(state_size, action_size)
model.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

env._max_episode_steps = maxSteps

epsilon = 1.0               # exploration rate
epsilon_min = 0.01
epsilon_decay = 0.999

step_history = []
reward_history = []

for iteration in range(iterationNum):
    tracker.start()

    state = env.reset()
    state = np.reshape(state, [1, state_size])
    rewardsum = 0

    try:
        # Epsilon-greedy selection
        for step in range(maxSteps):
            if step == 0 or np.random.rand(1) < epsilon:
                action = np.random.randint(0, action_size)
                print(f"\t[*] Random exploration. Selected action: {action}")
            else:
                action = np.argmax(model.predict(state)[0])
                print(f"\t[*] Exploiting gained knowledge. Selected action: {action}")

            # Step
            next_state, reward, done, _ = env.step(action)

            if done:
                print(f"iteration: {iteration}/{iterationNum}, step: {step}, rew: {rewardsum}, eps: {epsilon:.2}")
                break

            next_state = np.reshape(next_state, [1, state_size])

            # Train
            target = reward
            if not done:
                target = (reward + 0.95 * np.amax(model.predict(next_state)[0]))

            target_f = model.predict(state)
            target_f[0][action] = target
            model.fit(state, target_f, epochs=1, verbose=0)

            state = next_state
            rewardsum += reward
            if epsilon > epsilon_min: epsilon *= epsilon_decay

        step_history.append(step)
        reward_history.append(rewardsum)


    finally:
        print()
        tracker.stop()
        if iteration+1 == iterationNum:
            env.close()
            print("Done")
            break

print("Plot Learning Performance")
mpl.rcdefaults()
mpl.rcParams.update({'font.size': 16})

fig, ax = plt.subplots(figsize=(10,4))
plt.grid(True, linestyle='--')
plt.title('Learning Performance')
plt.plot(range(len(step_history)), step_history, label='Steps', marker="^", linestyle=":")#, color='red')
plt.plot(range(len(reward_history)), reward_history, label='Reward', marker="", linestyle="-")#, color='k')
plt.xlabel('Episode')
plt.ylabel('Time')
plt.legend(prop={'size': 12})

plt.savefig('learning_centralized.png', bbox_inches='tight')
plt.show()