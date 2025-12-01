# REINFORCE (Monte Carlo Policy Gradient)

## 1. Introduction
**REINFORCE** is the fundamental Policy Gradient algorithm. Unlike Q-Learning, which learns the value of states (Value-Based), REINFORCE learns the policy directly (Policy-Based). It optimizes the parameters of a policy network $\pi_\theta(a|s)$ to maximize the expected return.

## 2. Historical Context
*   **The Inventors:** The algorithm is credited to **Ronald Williams** (1992), published in his paper "Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning".
*   **Significance:** It laid the groundwork for modern policy gradient methods like PPO and TRPO, which power today's most advanced RL agents (including ChatGPT's RLHF phase).

## 3. Real-World Analogy
### The Trial-and-Error Coach
Imagine a coach teaching a player to shoot a basketball.
*   **Value-Based (Q-Learning):** The coach analyzes every spot on the court and assigns a score: "Standing here is worth 2 points."
*   **Policy-Based (REINFORCE):** The coach doesn't care about the spots. They just watch the player shoot.
    *   If the ball goes in (Positive Reward), the coach says: "Whatever you just did (muscle movement), **do it more**."
    *   If the ball misses (Negative Reward), the coach says: "Whatever you just did, **do it less**."
*   Crucially, the coach waits until the *end* of the sequence (Monte Carlo) to judge the outcome.

## 4. Mathematical Foundation

### The Objective
We want to maximize the expected return $J(\theta)$:
$$ J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} [R(\tau)] $$
where $\tau$ is a trajectory (sequence of states and actions).

### The Policy Gradient Theorem
The gradient of the objective is:
$$ \nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t | s_t) \cdot G_t \right] $$
*   $\nabla_\theta \log \pi_\theta(a_t | s_t)$: The direction to move parameters to make action $a_t$ more probable.
*   $G_t$: The return (cumulative reward) from time $t$. If $G_t$ is high, we push the policy strongly in that direction.

## 5. Algorithm Steps

1. Initialize Policy Network $\pi_\theta$.
2. For each episode:
   - Generate a full trajectory $\tau = (s_0, a_0, r_0, s_1, a_1, r_1, \dots)$ by following $\pi_\theta$.
   - For each step $t$ in the trajectory:
     - Calculate return $G_t = \sum_{k=t}^{T} \gamma^{k-t} r_k$.
     - Calculate loss: $L = - \log \pi_\theta(a_t | s_t) \cdot G_t$.
     - Accumulate gradients.
   - Update $\theta$ using Gradient Descent (or Adam).
