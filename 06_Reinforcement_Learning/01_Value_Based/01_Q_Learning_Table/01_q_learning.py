import numpy as np
import random
import os
import time

# Configuration
ROWS = 4
COLS = 4
START_STATE = (0, 0)
GOAL_STATE = (3, 3)
HOLES = [(1, 1), (1, 3), (2, 3), (3, 0)]

# Hyperparameters
ALPHA = 0.1       # Learning Rate
GAMMA = 0.99      # Discount Factor
EPSILON = 1.0     # Exploration Rate
EPSILON_DECAY = 0.995
EPSILON_MIN = 0.01
EPISODES = 1000
MAX_STEPS = 100

# Actions
ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT']
ACTION_MAP = {
    0: (-1, 0), # UP
    1: (1, 0),  # DOWN
    2: (0, -1), # LEFT
    3: (0, 1)   # RIGHT
}

class GridWorld:
    def __init__(self):
        self.rows = ROWS
        self.cols = COLS
        self.state = START_STATE
        
    def reset(self):
        self.state = START_STATE
        return self.state_to_index(self.state)
    
    def state_to_index(self, state):
        return state[0] * self.cols + state[1]
    
    def index_to_state(self, idx):
        return (idx // self.cols, idx % self.cols)
    
    def step(self, action_idx):
        move = ACTION_MAP[action_idx]
        new_r = max(0, min(self.rows - 1, self.state[0] + move[0]))
        new_c = max(0, min(self.cols - 1, self.state[1] + move[1]))
        new_state = (new_r, new_c)
        
        # Check outcome
        if new_state == GOAL_STATE:
            reward = 1.0
            done = True
        elif new_state in HOLES:
            reward = -1.0
            done = True
        else:
            reward = -0.01 # Small penalty to encourage shortest path
            done = False
            
        self.state = new_state
        return self.state_to_index(new_state), reward, done

def train():
    env = GridWorld()
    num_states = ROWS * COLS
    num_actions = len(ACTIONS)
    
    # Initialize Q-Table
    q_table = np.zeros((num_states, num_actions))
    
    print("Starting Q-Learning Training...")
    
    global EPSILON
    
    for episode in range(EPISODES):
        state_idx = env.reset()
        done = False
        total_reward = 0
        
        for _ in range(MAX_STEPS):
            # Epsilon-Greedy Action Selection
            if random.uniform(0, 1) < EPSILON:
                action_idx = random.randint(0, num_actions - 1) # Explore
            else:
                action_idx = np.argmax(q_table[state_idx])      # Exploit
                
            # Take action
            next_state_idx, reward, done = env.step(action_idx)
            
            # Bellman Update
            # Q(s,a) = Q(s,a) + alpha * (r + gamma * max(Q(s', a')) - Q(s,a))
            old_value = q_table[state_idx, action_idx]
            next_max = np.max(q_table[next_state_idx])
            
            new_value = old_value + ALPHA * (reward + GAMMA * next_max - old_value)
            q_table[state_idx, action_idx] = new_value
            
            state_idx = next_state_idx
            total_reward += reward
            
            if done:
                break
        
        # Decay Epsilon
        if EPSILON > EPSILON_MIN:
            EPSILON *= EPSILON_DECAY
            
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}: Total Reward: {total_reward:.2f}, Epsilon: {EPSILON:.2f}")
            
    return q_table

def visualize_policy(q_table):
    print("\nLearned Policy (Grid):")
    symbols = {0: '↑', 1: '↓', 2: '←', 3: '→'}
    
    for r in range(ROWS):
        row_str = "|"
        for c in range(COLS):
            state = (r, c)
            idx = r * COLS + c
            
            if state == GOAL_STATE:
                row_str += " G |"
            elif state in HOLES:
                row_str += " H |"
            else:
                best_action = np.argmax(q_table[idx])
                row_str += f" {symbols[best_action]} |"
        print(row_str)

if __name__ == "__main__":
    q_table = train()
    visualize_policy(q_table)
    
    # Save Q-Table
    os.makedirs("assets", exist_ok=True)
    np.save("assets/q_table.npy", q_table)
    print("\nQ-Table saved to assets/q_table.npy")
