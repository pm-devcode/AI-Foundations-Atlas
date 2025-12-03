# Q-Learning (Tabular)

## 1. Executive Summary
**Q-Learning** is a foundational model-free reinforcement learning algorithm. It enables an agent to learn the optimal action-selection policy for any given state by interacting with the environment. It works by learning a **Q-Function** (Quality Function) that estimates the expected total future reward for taking a specific action in a specific state. For simple environments with discrete states, this function is stored as a lookup table (Q-Table).

## 2. Historical Context
*   **Invention (1989)**: Introduced by **Chris Watkins** in his PhD thesis "Learning from Delayed Rewards".
*   **Significance**: It was a breakthrough because it proved that an agent could learn the optimal policy *off-policy* (learning from actions that are not necessarily the ones the current policy would choose) and without a model of the environment's physics (transition probabilities).

## 3. Real-World Analogy
**The Dog Training**
Imagine you are training a dog to "sit".
*   **The Agent**: The dog.
*   **The State**: You saying "Sit!".
*   **The Action**: The dog sits, jumps, or barks.
*   **The Reward**: If the dog sits, it gets a treat (+1). If it barks, it gets nothing (0) or a "No!" (-1).
*   **Q-Learning**: Over time, the dog learns that in the state "Command: Sit", the action "Sit" has the highest expected future reward (the treat). It doesn't know *why* (physics of sitting), but it knows the value of the action.

## 4. Mathematical Foundation
The core of the algorithm is the **Bellman Equation** update rule:

$$ Q^{new}(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \cdot \left( r_t + \gamma \cdot \max_{a} Q(s_{t+1}, a) - Q(s_t, a_t) \right) $$

Where:
*   $Q(s_t, a_t)$: Current estimated value of taking action $a$ in state $s$.
*   $\alpha$ (Alpha): Learning rate (0 to 1). How much we accept the new information.
*   $r_t$: Immediate reward received.
*   $\gamma$ (Gamma): Discount factor (0 to 1). How much we care about future rewards vs. immediate rewards.
*   $\max_{a} Q(s_{t+1}, a)$: The estimated value of the best action in the *next* state (our "target").

## 5. Architecture (The Q-Table)

For a GridWorld of size 4x4 with 4 possible actions (Up, Down, Left, Right), the Q-Table is a matrix of size 16x4.

```mermaid
graph TD
    State[State s] --> Lookup[Lookup in Q-Table]
    Lookup --> QValues[Q-Values for all Actions]
    QValues --> Policy{Policy (e.g., Epsilon-Greedy)}
    Policy -- Explore --> Random[Random Action]
    Policy -- Exploit --> Max[Action with Max Q-Value]
    
    style Lookup fill:#f9f,stroke:#333,stroke-width:2px
    style QValues fill:#ff9,stroke:#333,stroke-width:2px
```

## 6. Implementation Details
The repository contains a Python implementation (`01_q_learning.py`) for a custom **GridWorld** environment:

*   **Environment**: A 4x4 grid with a Start, a Goal (+1 reward), and Holes (-1 reward).
*   **Agent**: Uses a Q-Table to store values.
*   **Epsilon-Greedy Strategy**:
    *   With probability $\epsilon$, choose a random action (Explore).
    *   With probability $1-\epsilon$, choose the best action from the Q-Table (Exploit).
    *   $\epsilon$ decays over time, shifting the agent from exploration to exploitation.

## 7. How to Run
Run the script from the terminal:

```bash
python 01_q_learning.py
```

## 8. Implementation Results

### Learned Policy
After training for 1000 episodes, the agent learns to navigate around holes to reach the goal.

```text
Learned Policy (Grid):
| ↓ | ← | ↓ | ← |
| ↓ | H | ↓ | H |
| → | → | ↓ | H |
| H | → | → | G |
```
*(Arrows indicate the optimal action to take in each cell. H = Hole, G = Goal)*

## 9. References
*   Watkins, C. J. C. H. (1989). *Learning from Delayed Rewards*. PhD Thesis, Cambridge University.
*   Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction*.
