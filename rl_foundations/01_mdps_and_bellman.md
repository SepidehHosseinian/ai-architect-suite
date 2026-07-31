# 🧮 RL Math Foundation 01: Markov Decision Processes & Bellman Equations

## 1. Formal Definition of an MDP
A Markov Decision Process (MDP) is formally defined as a 5-tuple $\mathcal{M} = (\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma)$:

* $\mathcal{S}$: State space.
* $\mathcal{A}$: Action space.
* $\mathcal{P}(s' \mid s, a) = \mathbb{P}(S_{t+1} = s' \mid S_t = s, A_t = a)$: Transition probability function.
* $\mathcal{R}(s, a, s') = \mathbb{E}[R_{t+1} \mid S_t = s, A_t = a, S_{t+1} = s']$: Reward function.
* $\gamma \in [0, 1)$: Discount factor for future rewards.

### Markov Property
The state transitions satisfy the conditional independence property:

$$\mathbb{P}(S_{t+1} \mid S_t, A_t, S_{t-1}, A_{t-1}, \dots, S_0, A_0) = \mathbb{P}(S_{t+1} \mid S_t, A_t)$$

---

## 2. Value Functions & Expected Return
The discounted return $G_t$ at step $t$ is:

$$G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$$

### State-Value Function $V^\pi(s)$
The expected return starting from state $s$ following policy $\pi(a \mid s)$:

$$V^\pi(s) = \mathbb{E}_\pi \left[ G_t \;\middle\vert{}\; S_t = s \right]$$

### Action-Value Function $Q^\pi(s, a)$
The expected return starting from state $s$, taking action $a$, and thereafter following policy $\pi$:

$$Q^\pi(s, a) = \mathbb{E}_\pi \left[ G_t \;\middle\vert{}\; S_t = s, A_t = a \right]$$

---

## 3. Derivation of the Bellman Expectation Equation

Expanding $V^\pi(s)$ recursively:

$$\begin{aligned} V^\pi(s) &= \mathbb{E}_\pi \left[ R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \dots \;\middle\vert{}\; S_t = s \right] \\ &= \mathbb{E}_\pi \left[ R_{t+1} + \gamma G_{t+1} \;\middle\vert{}\; S_t = s \right] \\ &= \sum_{a \in \mathcal{A}} \pi(a \mid s) \sum_{s' \in \mathcal{S}} \mathcal{P}(s' \mid s, a) \left[ \mathcal{R}(s, a, s') + \gamma \mathbb{E}_\pi [G_{t+1} \mid S_{t+1} = s'] \right] \end{aligned}$$

$$\implies V^\pi(s) = \sum_{a \in \mathcal{A}} \pi(a \mid s) \sum_{s' \in \mathcal{S}} \mathcal{P}(s' \mid s, a) \left[ \mathcal{R}(s, a, s') + \gamma V^\pi(s') \right]$$

Similarly, for $Q^\pi(s, a)$:

$$Q^\pi(s, a) = \sum_{s' \in \mathcal{S}} \mathcal{P}(s' \mid s, a) \left[ \mathcal{R}(s, a, s') + \gamma \sum_{a' \in \mathcal{A}} \pi(a' \mid s') Q^\pi(s', a') \right]$$

---

## 4. Bellman Optimality Equations

For optimal value functions $V^*(s) = \max_\pi V^\pi(s)$ and $Q^*(s, a) = \max_\pi Q^\pi(s, a)$:

$$V^*(s) = \max_{a \in \mathcal{A}} \sum_{s' \in \mathcal{S}} \mathcal{P}(s' \mid s, a) \left[ \mathcal{R}(s, a, s') + \gamma V^*(s') \right]$$

$$Q^*(s, a) = \sum_{s' \in \mathcal{S}} \mathcal{P}(s' \mid s, a) \left[ \mathcal{R}(s, a, s') + \gamma \max_{a' \in \mathcal{A}} Q^*(s', a') \right]$$