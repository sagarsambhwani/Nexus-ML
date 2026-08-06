# Chapter 21: Reinforcement Learning, Bandits, and Sequential Decision Making

---

## 1. Big Picture

In supervised learning, models learn from passive static datasets with ground truth labels. In **Reinforcement Learning (RL)**, an agent actively interacts with a dynamic environment, taking sequential actions to maximize cumulative future rewards under uncertainty.

This chapter covers Multi-Armed Bandits ($\epsilon$-greedy, Upper Confidence Bound UCB, Thompson Sampling), Markov Decision Processes (MDP), Q-Learning, Deep Q-Networks (DQN), Policy Gradient (PPO), and Direct Preference Optimization (DPO for LLM alignment).

---

## 2. Intuition

- **Exploration vs Exploitation Trade-off**:
  - *Exploitation*: Ordering your favorite dish at your favorite restaurant (guaranteed good reward, zero learning).
  - *Exploration*: Trying a brand new dish at a new restaurant (risk of terrible dish, potential discovery of your new lifetime favorite food!).
- **Markov Decision Process (MDP)**: The world satisfies the Markov Property if the future state depends *only* on the current state $s_t$ and action $a_t$, not on historical path trajectory $s_0 \dots s_{t-1}$.

---

## 3. Visualization

```text
Reinforcement Learning Feedback Loop:

                  ┌────────────────────────┐
                  │      ENVIRONMENT       │
                  └────────────────────────┘
                    ▲                    │
     Action a_t     │                    │ State s_t, Reward r_t
                    │                    ▼
                  ┌────────────────────────┐
                  │         AGENT          │
                  │ (Policy π / Value Q)   │
                  └────────────────────────┘
```

---

## 4. Mathematics

### 1. Upper Confidence Bound (UCB1 Algorithm)
Action selection formula balancing empirical mean reward $\hat{Q}(a)$ with uncertainty exploration bonus:

$$a_t = \arg\max_{a \in A} \left[ \hat{Q}(a) + c \cdot \sqrt{\frac{\ln t}{N_a(t)}} \right]$$

Where $t$ is total step count, $N_a(t)$ is count of times action $a$ was selected, and $c$ controls exploration weight.

### 2. Bellman Optimality Equation
For optimal Action-Value Function $Q^*(s, a)$:

$$Q^*(s, a) = \mathbb{E} \left[ R(s, a) + \gamma \max_{a'} Q^*(s', a') \;\middle|\; s_t = s, a_t = a \right]$$

Temporal Difference (TD) Error update:
$$\text{TD\_Error} = \delta_t = r_t + \gamma \max_{a'} Q(s', a') - Q(s, a)$$

---

## 5. Python (From Scratch Multi-Armed Bandit UCB)

```python
import numpy as np

class MultiArmedBanditUCB:
    def __init__(self, n_arms=4, c=1.5):
        self.n_arms = n_arms
        self.c = c
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)
        
    def select_action(self, t):
        # Pull each arm once initially
        for arm in range(self.n_arms):
            if self.counts[arm] == 0:
                return arm
                
        # UCB1 Selection Formula
        ucb_values = self.values + self.c * np.sqrt(np.log(t + 1) / self.counts)
        return int(np.argmax(ucb_values))
        
    def update(self, arm, reward):
        self.counts[arm] += 1
        # Incremental Mean Update Rule: Q_new = Q_old + (1/n)*(R - Q_old)
        self.values[arm] += (reward - self.values[arm]) / self.counts[arm]

# Test Bandit Environment (True Win Probabilities: Arm 2 is best!)
true_rewards = [0.1, 0.3, 0.7, 0.2]
bandit = MultiArmedBanditUCB(n_arms=4)

for t in range(1000):
    arm = bandit.select_action(t)
    reward = 1.0 if np.random.rand() < true_rewards[arm] else 0.0
    bandit.update(arm, reward)

print("Pull Counts per Arm:", bandit.counts)
print("Learned Estimated Expected Rewards:", np.round(bandit.values, 3))
```

---

## 6. Production Library (PyTorch DQN Agent)

```python
import torch
import torch.nn as nn

# Production Deep Q-Network (DQN) Architecture
class QNetwork(nn.Module):
    def __init__(self, state_dim=8, action_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
        
    def forward(self, state):
        return self.net(state)

dqn = QNetwork()
sample_state = torch.randn(1, 8)
q_values = dqn(sample_state)
print("Q-Values across 4 candidate actions:", q_values.detach().numpy())
```

---

## 7. Under the Hood

- **DQN Stability Innovations**:
  1. *Experience Replay Buffer*: Sampling transitions $(s, a, r, s')$ uniformly breaks temporal correlations between consecutive training batches.
  2. *Target Network*: Maintaining a separate, slowly updated target network $\theta^-$ prevents self-referential training instability when computing target value $r + \gamma \max Q(s', a'; \theta^-)$.

---

## 8. Engineering Perspective

- **Bandits in Production**: Multi-Armed Bandits (UCB / Thompson Sampling) outperform traditional static A/B testing by dynamically routing web traffic to winning variants in real-time while minimizing wasted traffic on poor variants.

---

## 9. Common Mistakes

1. **Forgetting Discount Factor $\gamma$**: Setting $\gamma = 1.0$ in infinite-horizon environments causes state values to diverge to infinity. Use $\gamma \in [0.95, 0.99]$.
2. **Training Policy Gradients Without Advantage Normalization**: Raw policy gradients exhibit extreme variance. Normalize advantage estimates $\hat{A}_t = \frac{A_t - \mu}{\sigma + 1e-8}$.

---

## 10. Interview Questions

### Q1: Compare Q-Learning (Off-Policy) vs SARSA (On-Policy).
**Answer**: Q-Learning is an **off-policy** algorithm that updates $Q(s, a)$ assuming the agent takes the greedy optimal action in the next state ($\max_{a'} Q(s', a')$), regardless of the actual action executed. SARSA is an **on-policy** algorithm that updates $Q(s, a)$ using the actual action $a'$ selected by the current exploration policy ($\pi(s')$).

---

## 11. Exercises

1. **Math**: Prove the Convergence of Incremental Sample Average Update Rule $Q_{n+1} = Q_n + \frac{1}{n}(R_n - Q_n)$.
2. **Coding**: Implement Thompson Sampling using Beta distributions in Python for A/B testing optimization.

---

## 12. Mini Project: Deep Q-Learning Game Agent

Write a script `dqn_agent.py` using PyTorch that trains a DQN agent with Experience Replay to solve the OpenAI Gymnasium `CartPole-v1` environment.

---

## 13. Capstone Integration

Powers real-time exploration algorithms in `src/recommendation/pipeline.py`.
