import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os

# Configuration
ENV_NAME = "CartPole-v1"
GAMMA = 0.99
LR = 1e-2
EPISODES = 500
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create assets directory
os.makedirs("assets", exist_ok=True)

# -----------------------------------------------------------------------------
# 1. Policy Network
# -----------------------------------------------------------------------------
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=1) # Output probabilities for actions
        )
        
    def forward(self, x):
        return self.net(x)

# -----------------------------------------------------------------------------
# 2. REINFORCE Agent
# -----------------------------------------------------------------------------
class REINFORCEAgent:
    def __init__(self, state_dim, action_dim):
        self.policy = PolicyNetwork(state_dim, action_dim).to(DEVICE)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=LR)
        self.log_probs = []
        self.rewards = []
        
    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        probs = self.policy(state)
        
        # Sample action from probability distribution
        m = torch.distributions.Categorical(probs)
        action = m.sample()
        
        # Store log probability for gradient calculation later
        self.log_probs.append(m.log_prob(action))
        
        return action.item()
    
    def store_reward(self, reward):
        self.rewards.append(reward)
        
    def update(self):
        R = 0
        returns = []
        
        # Calculate discounted returns (Monte Carlo) in reverse
        for r in self.rewards[::-1]:
            R = r + GAMMA * R
            returns.insert(0, R)
            
        returns = torch.tensor(returns).to(DEVICE)
        
        # Normalize returns (stabilizes training)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        
        policy_loss = []
        for log_prob, R in zip(self.log_probs, returns):
            # Loss = - log(prob) * Return
            # We want to maximize Reward, so we minimize negative Reward
            policy_loss.append(-log_prob * R)
            
        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()
        
        # Clear memory
        self.log_probs = []
        self.rewards = []

# -----------------------------------------------------------------------------
# 3. Training Loop
# -----------------------------------------------------------------------------
def train():
    env = gym.make(ENV_NAME)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = REINFORCEAgent(state_dim, action_dim)
    
    print(f"Starting REINFORCE Training on {ENV_NAME} using {DEVICE}...")
    
    rewards_history = []
    
    for episode in range(EPISODES):
        state, _ = env.reset()
        total_reward = 0
        done = False
        truncated = False
        
        while not (done or truncated):
            action = agent.select_action(state)
            next_state, reward, done, truncated, _ = env.step(action)
            
            agent.store_reward(reward)
            state = next_state
            total_reward += reward
            
        agent.update()
        rewards_history.append(total_reward)
        
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(rewards_history[-10:])
            print(f"Episode {episode+1}/{EPISODES} | Avg Reward: {avg_reward:.2f}")
            
            if avg_reward > 475:
                print("Solved!")
                break
                
    env.close()
    return rewards_history, agent

def plot_results(rewards):
    plt.figure(figsize=(10, 5))
    plt.plot(rewards)
    plt.title("REINFORCE Training Rewards (CartPole-v1)")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.savefig("assets/reinforce_cartpole.png")
    print("Saved training plot to assets/reinforce_cartpole.png")

if __name__ == "__main__":
    rewards, agent = train()
    plot_results(rewards)
    
    torch.save(agent.policy.state_dict(), "assets/reinforce_cartpole.pth")
    print("Saved model to assets/reinforce_cartpole.pth")
