# Q-Learning (Tabular)

## 1. Introduction
Q-Learning is a model-free reinforcement learning algorithm to learn the value of an action in a particular state. It does not require a model of the environment (hence "model-free"), and it can handle problems with stochastic transitions and rewards without requiring adaptations.

## 2. Historical Context
*   **The Inventor:** Introduced by **Chris Watkins** in his 1989 PhD thesis, "Learning from Delayed Rewards".
*   **Significance:** It was a breakthrough because it proved that an agent could learn the optimal policy without modeling the environment's transition probabilities, bridging the gap between dynamic programming and trial-and-error learning.

## 3. Real-World Analogy
### The Dog Training
Imagine you are training a dog to "sit".
*   **The Agent:** The dog.
*   **The State:** You saying "Sit!".
*   **The Action:** The dog sits, jumps, or barks.
*   **The Reward:** If the dog sits, it gets a treat (+1). If it barks, it gets nothing (0) or a "No!" (-1).
*   **Q-Learning:** Over time, the dog learns that in the state "Command: Sit", the action "Sit" has the highest expected future reward (the treat). It doesn't know *why* (physics of sitting), but it knows the value of the action.

## 4. Core Concepts

### 4.1 The Q-Table
The core of the algorithm is the **Q-Table**, a matrix where:
- **Rows** represent States ($S$).
- **Columns** represent Actions ($A$).
- Each cell $Q(s, a)$ stores the expected future reward for taking action $a$ in state $s$.

### 2.2 The Bellman Equation
The Q-values are updated using the Bellman equation:

$$ Q^{new}(s_t, a_t) \leftarrow \underbrace{Q(s_t, a_t)}_{\text{current value}} + \underbrace{\alpha}_{\text{learning rate}} \cdot \left( \underbrace{r_t}_{\text{reward}} + \underbrace{\gamma}_{\text{discount factor}} \cdot \underbrace{\max_{a} Q(s_{t+1}, a)}_{\text{estimate of optimal future value}} - \underbrace{Q(s_t, a_t)}_{\text{current value}} \right) $$

Where:
- $\alpha$ (Alpha): Learning rate (0 to 1). How much we override old information.
- $\gamma$ (Gamma): Discount factor (0 to 1). Importance of future rewards.
- $r_t$: Reward received after taking action $a_t$.

### 2.3 Exploration vs. Exploitation ($\epsilon$-Greedy)
To learn effectively, the agent must balance:
- **Exploration**: Trying random actions to discover new states ($\epsilon$ probability).
- **Exploitation**: Choosing the best known action from the Q-Table ($1 - \epsilon$ probability).

## 3. Algorithm Steps

1. Initialize Q-Table with zeros.
2. For each episode:
   - Reset state $S$.
   - Loop until terminal state:
     - Choose action $A$ using $\epsilon$-greedy policy.
     - Take action $A$, observe reward $R$ and new state $S'$.
     - Update $Q(S, A)$ using the Bellman equation.
     - Set $S \leftarrow S'$.
3. Decay $\epsilon$ to reduce exploration over time.
