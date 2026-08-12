# 🎮 Chapter 16: Reinforcement Learning & Sequential Decision Making

## 16.1 The Reinforcement Learning Paradigm
Unlike Supervised Learning (learning from labeled targets) or Unsupervised Learning (discovering latent patterns), **Reinforcement Learning (RL)** models an agent learning to make sequential decisions by interacting with an environment to maximize cumulative reward.

```
                             Agent-Environment Interaction Loop
                                     Action a_t
                             ┌────────────────────────┐
                             ▼                        │
                       ┌───────────┐            ┌─────┴─────┐
                       │           │            │           │
                       │Environment│            │   Agent   │
                       │           │            │           │
                       └─────┬─────┘            └─────▲─────┘
                             │                        │
                             └────────────────────────┘
                                State s_t, Reward r_t
```

---

## 16.2 Markov Decision Processes (MDP)

An MDP is defined by 5 tuples $(\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma)$:
- $\mathcal{S}$: Set of environment states.
- $\mathcal{A}$: Set of agent actions.
- $\mathcal{P}(s' \mid s, a)$: State transition probability $P(S_{t+1} = s' \mid S_t = s, A_t = a)$.
- $\mathcal{R}(s, a, s')$: Reward function $R_{t+1}$ received after transition.
- $\gamma \in [0, 1)$: Discount factor prioritizing immediate vs future rewards.

### Cumulative Discounted Return:
$$G_t = \sum_{k=0}^\infty \gamma^k R_{t+k+1}$$

---

## 16.3 Value Functions & Bellman Equations

### 1. State-Value Function $V^\pi(s)$
Expected return starting from state $s$ following policy $\pi$:

$$V^\pi(s) = \mathbb{E}_\pi \left[ G_t \mid S_t = s \right]$$

### 2. Action-Value Function $Q^\pi(s, a)$
Expected return starting from state $s$, taking action $a$, and following policy $\pi$:

$$Q^\pi(s, a) = \mathbb{E}_\pi \left[ G_t \mid S_t = s, A_t = a \right]$$

### Bellman Optimality Equations:
$$V^*(s) = \max_{a \in \mathcal{A}} \sum_{s', r} \mathcal{P}(s', r \mid s, a) \left[ r + \gamma V^*(s') \right]$$
$$Q^*(s, a) = \sum_{s', r} \mathcal{P}(s', r \mid s, a) \left[ r + \gamma \max_{a'} Q^*(s', a') \right]$$

---

## 16.4 Q-Learning & Deep Q-Networks (DQN)

### 1. Q-Learning (Off-Policy Temporal Difference)
Updates action values $Q(s, a)$ iteratively using Temporal Difference (TD) error:

$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ \underbrace{r + \gamma \max_{a'} Q(s', a')}_{\text{TD Target}} - \underbrace{Q(s, a)}_{\text{Current Estimate}} \right]$$

### 2. Deep Q-Networks (DQN)
Deep Q-Networks approximate high-dimensional action-value functions $Q(s, a; \theta) \approx Q^*(s, a)$ using Deep Neural Networks.

```
                        DQN Stability Innovations
                                    │
           ┌────────────────────────┴────────────────────────┐
           ▼                                                 ▼
   Experience Replay                                Target Network (θ⁻)
• Stores transitions (s, a, r, s')               • Frozen parameters updated periodically
• Random batch sampling breaks correlation        • Prevents target moving instability
```

#### DQN Loss Function:
$$\mathcal{L}(\theta) = \mathbb{E}_{(s, a, r, s') \sim U(D)} \left[ \left( r + \gamma \max_{a'} Q(s', a'; \theta^-) - Q(s, a; \theta) \right)^2 \right]$$

---

## 16.5 Policy Gradient Methods & PPO / DPO

Rather than estimating value functions $Q(s, a)$, Policy Gradient methods parameterize policy $\pi_\theta(a \mid s)$ directly.

### REINFORCE Policy Gradient Theorem:
$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \nabla_\theta \log \pi_\theta(a_t \mid s_t) G_t \right]$$

### Proximal Policy Optimization (PPO):
PPO introduces a clipped surrogate objective function preventing destructive large policy updates:

$$L^{\text{CLIP}}(\theta) = \hat{\mathbb{E}}_t \left[ \min\left( r_t(\theta) \hat{A}_t, \; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]$$

where probability ratio is $r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}$, and $\hat{A}_t$ is the Advantage function estimate.

### Direct Preference Optimization (DPO):
Used in aligning Large Language Models (LLMs) with human feedback (RLHF) without requiring a separate complex reward model:

$$\mathcal{L}_{\text{DPO}}(\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right) \right]$$

---

## ⚓ Repository Code Reference
- See [`src/recommendation/pipeline.py`](file:///e:/Downloads/ML_only/src/recommendation/pipeline.py) for sequential recommendation affinity scoring.
