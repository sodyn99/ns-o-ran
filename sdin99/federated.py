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
import tensorflow_federated as tff
import nest_asyncio
nest_asyncio.apply()
import numpy as np
# from codecarbon import track_emissions
from codecarbon import EmissionsTracker
from comet_ml import Experiment
import random

__author__ = "Lee Sungjin"
__copyright__ = "Copyright (c) 2024, Lee Sungjin"
__version__ = "0.1.0"
__email__ = "ssjj3552@gmail.com"

tracker = EmissionsTracker(
    project_name="federated",
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

# TFF 실행 환경 설정
tff.backends.native.set_local_execution_context()

# 클라이언트별 환경 및 모델 설정
def client_environment_setup(client_id):
    client_port = random.randrange(8000,9000) # port + client_id
    env = ns3env.Ns3Env(port=client_port, startSim=startSim, simSeed=client_id)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    return env, state_size, action_size

def create_keras_model(state_size, action_size):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(state_size, activation='relu', input_shape=(state_size,)),
        tf.keras.layers.Dense(action_size, activation='softmax')
    ])
    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()
    return model

def client_update(model, dataset, max_steps, state_size, action_size, epsilon=1.0, epsilon_decay=0.999, epsilon_min=0.01):
    reward_history = []
    step_history = []

    for state, action, next_state, reward, done in dataset:

        state = np.reshape(state, [1, state_size])

        if np.random.rand() < epsilon:
            action = np.random.randint(0, action_size)
            print(f"\t[*] Random exploration. Selected action: {action}")
        else:
            action = np.argmax(model.predict(state.reshape(1, -1))[0])
            print(f"\t[*] Exploiting gained knowledge. Selected action: {action}")

        next_state = np.reshape(next_state, [1, state_size])

        # Update the model
        target = reward
        if not done:
            target = (reward + 0.95 * np.amax(model.predict(next_state)[0]))

        target_vec = model.predict(state)[0]
        target_vec[action] = target
        model.fit(state.reshape(1, -1), target_vec.reshape(-1, action_size), epochs=1, verbose=0)

        # Decay epsilon
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        reward_history.append(reward)
        step_history.append(action)

    return model.get_weights(), reward_history, step_history

def generate_client_data(env, num_samples=100):
    data = []
    state = env.reset()
    for _ in range(num_samples):
        action = np.random.randint(0, env.action_space.n)  # 무작위 행동 선택
        next_state, reward, done, _ = env.step(action)
        data.append((state, action, next_state, reward, done))
        state = next_state if not done else env.reset()
    return data

def federated_train(client_ids, num_iterations=iterationNum):
    initial_client = 0
    env, state_size, action_size = client_environment_setup(initial_client)
    global_model = create_keras_model(state_size, action_size)

    for iteration in range(num_iterations):
        tracker.start()
        local_models = []
        reward_history = []
        step_history = []

        for client_id in client_ids:
            env, state_size, action_size = client_environment_setup(client_id)
            local_data = generate_client_data(env)
            model_weights, rewards, steps = client_update(global_model, local_data, max_steps=maxSteps, state_size=state_size, action_size=action_size)
            local_models.append(model_weights)
            reward_history.append(rewards)
            step_history.append(steps)

        # 모든 로컬 모델 가중치에 대해 층별로 평균 계산
        new_weights = []
        # 가중치 리스트의 길이는 첫 번째 로컬 모델의 층 수와 동일
        for layer in range(len(local_models[0])):
            # 층별로 모든 로컬 모델의 가중치를 가져와 평균을 계산
            layer_weights = np.array([model[layer] for model in local_models])
            weighted_average = np.mean(layer_weights, axis=0)
            new_weights.append(weighted_average)

        global_model.set_weights(new_weights)

        # Log performance
        print(f"Episode {iteration + 1}, Average Rewards: {np.mean(reward_history)}")

        tracker.stop()

    return global_model, reward_history, step_history

# 훈련 실행
clients = [1] # [1, 2, 3]
federated_model, reward_history, step_history = federated_train(clients)

# 성능 시각화
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

plt.savefig('learning_federated.png', bbox_inches='tight')
plt.show()