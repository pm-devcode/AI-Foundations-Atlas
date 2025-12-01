# Deep Q-Network (DQN)

## 1. Introduction
Deep Q-Network (DQN) combines Q-Learning with deep neural networks to handle high-dimensional state spaces (like pixels in Atari games or continuous states in robotics).

## 2. Historical Context
*   **The Inventors:** Developed by **DeepMind** (Volodymyr Mnih et al.) in 2013, with the seminal Nature paper published in 2015.
*   **The Breakthrough:** It was the first artificial agent capable of learning to excel at a diverse set of challenging tasks (Atari 2600 games) solely from raw pixel inputs, achieving human-level performance. This marked the beginning of the "Deep Reinforcement Learning" era.

## 3. Real-World Analogy
### The Gamer Watching Replays
Imagine a professional gamer learning a new game.
*   **Playing (Exploration):** Initially, they play randomly to see what happens.
*   **Replay Buffer (Experience Replay):** Instead of forgetting a game immediately after playing, they record it. Later, they watch recordings of their past games—both wins and losses—to analyze what went right or wrong. This prevents them from forgetting old strategies while learning new ones (breaking correlation).
*   **Target Network:** It's like having a "Coach" who sets the standard. The Coach doesn't change their mind every second; they keep the criteria stable for a while, then update their standards once the player improves.

## 4. Core Concepts

### 4.1 Function Approximation
Instead of a Q-Table, we use a Neural Network $Q(s, a; \theta)$ to approximate the Q-values.
- **Input**: State $s$.
- **Output**: Q-values for all possible actions $a$.

### 2.2 Experience Replay
Training a neural network on consecutive samples is unstable because they are highly correlated. DQN stores transitions $(s, a, r, s', done)$ in a **Replay Buffer** and samples random mini-batches for training. This breaks correlations and stabilizes training.

### 2.3 Target Network
Using the same network to calculate the target value and the predicted value leads to oscillation. DQN uses a separate **Target Network** $\hat{Q}$ with parameters $\theta^-$ that are updated slowly (or periodically) to match the main network $\theta$.

The loss function becomes:
$$ L(\theta) = \mathbb{E} \left[ \left( r + \gamma \max_{a'} \hat{Q}(s', a'; \theta^-) - Q(s, a; \theta) \right)^2 \right] $$

## 3. Algorithm Steps

1. Initialize Main Network $Q$ and Target Network $\hat{Q}$.
2. Initialize Replay Buffer.
3. For each step in environment:
   - Select action using $\epsilon$-greedy policy.
   - Execute action, observe reward $r$ and next state $s'$.
   - Store transition $(s, a, r, s')$ in Replay Buffer.
   - Sample random batch from Replay Buffer.
   - Compute targets: $y = r + \gamma \max \hat{Q}(s', a')$.
   - Update Main Network $Q$ by minimizing loss between $y$ and $Q(s, a)$.
   - Periodically update Target Network $\hat{Q} \leftarrow Q$.
