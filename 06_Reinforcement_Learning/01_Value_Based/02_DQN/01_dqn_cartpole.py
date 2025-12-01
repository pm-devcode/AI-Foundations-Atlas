import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import os
import matplotlib.pyplot as plt

# Configuration
ENV_NAME = "CartPole-v1"
GAMMA = 0.99
BATCH_SIZE = 64
LR = 1e-3
MEMORY_SIZE = 10000
MIN_MEMORY_SIZE = 1000
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
TARGET_UPDATE_FREQ = 10
EPISODES = 300
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create assets directory
os.makedirs("assets", exist_ok=True)

# -----------------------------------------------------------------------------
# 1. Replay Buffer
# -----------------------------------------------------------------------------
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (
            np.array(state),
            np.array(action),
            np.array(reward),
            np.array(next_state),
            np.array(done)
        )
    
    def __len__(self):
        return len(self.buffer)

# -----------------------------------------------------------------------------
# 2. Neural Network (DQN)
# -----------------------------------------------------------------------------
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
        
    def forward(self, x):
        return self.net(x)

# -----------------------------------------------------------------------------
# 3. Agent
# -----------------------------------------------------------------------------
class Agent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.epsilon = EPSILON_START
        
        self.policy_net = DQN(state_dim, action_dim).to(DEVICE)
        self.target_net = DQN(state_dim, action_dim).to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        self.memory = ReplayBuffer(MEMORY_SIZE)
        self.steps_done = 0
        
    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
                q_values = self.policy_net(state_t)
                return q_values.argmax().item()
                
    def update(self):
        if len(self.memory) < MIN_MEMORY_SIZE:
            return
        
        states, actions, rewards, next_states, dones = self.memory.sample(BATCH_SIZE)
        
        states = torch.FloatTensor(states).to(DEVICE)
        actions = torch.LongTensor(actions).unsqueeze(1).to(DEVICE)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(DEVICE)
        next_states = torch.FloatTensor(next_states).to(DEVICE)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(DEVICE)
        
        # Q(s, a)
        current_q = self.policy_net(states).gather(1, actions)
        
        # Max Q(s', a') from target net
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target_q = rewards + (GAMMA * next_q * (1 - dones))
            
        loss = nn.MSELoss()(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Epsilon Decay
        if self.epsilon > EPSILON_END:
            self.epsilon *= EPSILON_DECAY
            
        self.steps_done += 1
        if self.steps_done % TARGET_UPDATE_FREQ == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

# -----------------------------------------------------------------------------
# 4. Training Loop
# -----------------------------------------------------------------------------
def train():
    env = gym.make(ENV_NAME)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = Agent(state_dim, action_dim)
    
    print(f"Starting DQN Training on {ENV_NAME} using {DEVICE}...")
    
    rewards_history = []
    
    for episode in range(EPISODES):
        state, _ = env.reset()
        total_reward = 0
        done = False
        truncated = False
        
        while not (done or truncated):
            action = agent.select_action(state)
            next_state, reward, done, truncated, _ = env.step(action)
            
            # Modify reward for faster convergence (optional but helpful for CartPole)
            # CartPole gives +1 for every step.
            
            agent.memory.push(state, action, reward, next_state, done or truncated)
            agent.update()
            
            state = next_state
            total_reward += reward
            
        rewards_history.append(total_reward)
        
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(rewards_history[-10:])
            print(f"Episode {episode+1}/{EPISODES} | Avg Reward: {avg_reward:.2f} | Epsilon: {agent.epsilon:.2f}")
            
            # Early stopping if solved
            if avg_reward > 475:
                print("Solved!")
                break
                
    env.close()
    return rewards_history, agent

def plot_results(rewards):
    plt.figure(figsize=(10, 5))
    plt.plot(rewards)
    plt.title("DQN Training Rewards (CartPole-v1)")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.savefig("assets/dqn_cartpole.png")
    print("Saved training plot to assets/dqn_cartpole.png")

if __name__ == "__main__":
    rewards, agent = train()
    plot_results(rewards)
    
    # Save model
    torch.save(agent.policy_net.state_dict(), "assets/dqn_cartpole.pth")
    print("Saved model to assets/dqn_cartpole.pth")
