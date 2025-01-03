import gym
import cv2
import time
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque

# Environment settings
ENVIRONMENT = "PongDeterministic-v4"
DEVICE = torch.device("cpu")
SAVE_MODELS = True
MODEL_PATH = "./pong_model/"
SAVE_MODEL_INTERVAL = 10
TRAIN_MODEL = True
LOAD_MODEL_FROM_FILE = False
LOAD_FILE_EPISODE = 900

# Hyperparameters
BATCH_SIZE = 64
MAX_EPISODE = 1000
MAX_STEP = 100000
MAX_MEMORY_LEN = 50000
MIN_MEMORY_LEN = 40000
GAMMA = 0.97
ALPHA = 0.00025
EPSILON_DECAY = 0.99
RENDER_GAME_WINDOW = False

class DuelCNN(nn.Module):
    def __init__(self, h, w, output_size):
        super(DuelCNN, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        convw, convh = self.conv2d_size_calc(w, h, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        convw, convh = self.conv2d_size_calc(convw, convh, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)
        convw, convh = self.conv2d_size_calc(convw, convh, kernel_size=3, stride=1)
        linear_input_size = convw * convh * 64

        # Advantage stream
        self.Alinear1 = nn.Linear(linear_input_size, 128)
        self.Alrelu = nn.LeakyReLU()
        self.Alinear2 = nn.Linear(128, output_size)

        # Value stream
        self.Vlinear1 = nn.Linear(linear_input_size, 128)
        self.Vlrelu = nn.LeakyReLU()
        self.Vlinear2 = nn.Linear(128, 1)

    def conv2d_size_calc(self, w, h, kernel_size=5, stride=2):
        next_w = (w - kernel_size + 1) // stride
        next_h = (h - kernel_size + 1) // stride
        return next_w, next_h

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        Ax = self.Alrelu(self.Alinear1(x))
        Ax = self.Alinear2(Ax)
        Vx = self.Vlrelu(self.Vlinear1(x))
        Vx = self.Vlinear2(Vx)
        q = Vx + (Ax - Ax.mean(dim=1, keepdim=True))
        return q

class Agent:
    def __init__(self, environment):
        self.state_size_h, self.state_size_w, self.state_size_c = environment.observation_space.shape
        self.action_size = environment.action_space.n
        self.target_h, self.target_w = 80, 64
        self.crop_dim = [20, self.state_size_h, 0, self.state_size_w]
        self.gamma = GAMMA
        self.alpha = ALPHA
        self.epsilon = 1
        self.epsilon_decay = EPSILON_DECAY
        self.epsilon_minimum = 0.05
        self.memory = deque(maxlen=MAX_MEMORY_LEN)
        self.online_model = DuelCNN(self.target_h, self.target_w, self.action_size).to(DEVICE)
        self.target_model = DuelCNN(self.target_h, self.target_w, self.action_size).to(DEVICE)
        self.target_model.load_state_dict(self.online_model.state_dict())
        self.target_model.eval()
        self.optimizer = optim.Adam(self.online_model.parameters(), lr=self.alpha)

    def preProcess(self, image):
        frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        frame = frame[self.crop_dim[0]:self.crop_dim[1], self.crop_dim[2]:self.crop_dim[3]]
        frame = cv2.resize(frame, (self.target_w, self.target_h))
        frame = frame / 255.0
        return frame

    def act(self, state):
        if random.uniform(0, 1) <= self.epsilon:
            return random.randrange(self.action_size)
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float, device=DEVICE).unsqueeze(0)
            return torch.argmax(self.online_model(state_tensor)).item()

    def train(self):
        if len(self.memory) < MIN_MEMORY_LEN:
            return 0, 0
        state, action, reward, next_state, done = zip(*random.sample(self.memory, BATCH_SIZE))
        state = torch.tensor(np.concatenate(state), dtype=torch.float, device=DEVICE)
        next_state = torch.tensor(np.concatenate(next_state), dtype=torch.float, device=DEVICE)
        action = torch.tensor(action, dtype=torch.long, device=DEVICE)
        reward = torch.tensor(reward, dtype=torch.float, device=DEVICE)
        done = torch.tensor(done, dtype=torch.float, device=DEVICE)

        state_q_values = self.online_model(state)
        next_q_values = self.online_model(next_state)
        next_target_q_values = self.target_model(next_state)

        selected_q_value = state_q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        next_target_q_value = next_target_q_values.gather(
            1, next_q_values.max(1)[1].unsqueeze(1)
        ).squeeze(1)
        expected_q_value = reward + self.gamma * next_target_q_value * (1 - done)
        loss = F.mse_loss(selected_q_value, expected_q_value.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item(), torch.max(state_q_values).item()

    def storeResults(self, state, action, reward, next_state, done):
        self.memory.append([state[None, :], action, reward, next_state[None, :], done])

    def adaptiveEpsilon(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_minimum)


if __name__ == "__main__":
    environment = gym.make(ENVIRONMENT, render_mode='rgb_array')
    agent = Agent(environment)
    start_episode = 1

    if LOAD_MODEL_FROM_FILE:
        agent.online_model.load_state_dict(torch.load(f"{MODEL_PATH}{LOAD_FILE_EPISODE}.pkl"))
        with open(f"{MODEL_PATH}{LOAD_FILE_EPISODE}.json") as f:
            agent.epsilon = json.load(f)['epsilon']
        start_episode = LOAD_FILE_EPISODE + 1

    for episode in range(start_episode, MAX_EPISODE):
        state, _ = environment.reset()
        state = agent.preProcess(state)
        state = np.stack([state] * 4)
        total_reward, total_loss, total_max_q = 0, 0, 0
        for step in range(MAX_STEP):
            if RENDER_GAME_WINDOW:
                environment.render()
            action = agent.act(state)
            next_state, reward, done, _, _ = environment.step(action)
            next_state = agent.preProcess(next_state)
            next_state = np.stack([next_state, state[0], state[1], state[2]])
            agent.storeResults(state, action, reward, next_state, done)
            state = next_state

            if TRAIN_MODEL:
                loss, max_q = agent.train()
            else:
                loss, max_q = 0, 0

            total_loss += loss
            total_max_q += max_q
            total_reward += reward

            if step % 1000 == 0:
                agent.adaptiveEpsilon()

            if done:
                print(f"Episode: {episode}, Reward: {total_reward}, Loss: {total_loss}")
                break
