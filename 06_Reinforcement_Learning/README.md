# Reinforcement Learning

## Overview
Reinforcement Learning (RL) is about learning what to do—how to map situations to actions—so as to maximize a numerical reward signal. The learner is not told which actions to take, but instead must discover which actions yield the most reward by trying them.

## Key Categories

### Policy Based Methods
Learning the policy (mapping from state to action) directly.
*   **REINFORCE**: A Monte-Carlo Policy Gradient method that updates policy parameters in the direction of the gradient of expected reward.

### Value Based Methods
Learning the value of being in a given state or taking a specific action.
*   **DQN (Deep Q-Network)**: Using a neural network to approximate the Q-function, enabling RL in high-dimensional state spaces (like video game pixels).
*   **Q-Learning**: A tabular method for learning the optimal action-value function.
