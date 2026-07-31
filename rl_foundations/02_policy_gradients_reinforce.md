# 🧮 RL Math Foundation 02: Policy Gradient Theorem & REINFORCE

## 1. Objective Function
Let $\pi_\theta(a \mid s)$ be a parameterized policy. The objective $J(\theta)$ is the expected return over trajectories $\tau = (s_0, a_0, s_1, a_1, \dots, s_T)$:

$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} [R(\tau)] = \int P(\tau; \theta) R(\tau) d\tau$$

Where the trajectory probability $P(\tau; \theta)$ is given by:

$$P(\tau; \theta) = \mu(s_0) \prod_{t=0}^{T-1} \pi_\theta(a_t \mid s_t) \mathcal{P}(s_{t+1} \mid s_t, a_t)$$

---

## 2. Derivation of the Policy Gradient Theorem

We compute the gradient $\nabla_\theta J(\theta)$ using the log-derivative trick ($\nabla_\theta P = P \nabla_\theta \log P$):

$$\begin{aligned} \nabla_\theta J(\theta) &= \nabla_\theta \int P(\tau; \theta) R(\tau) d\tau \\ &= \int \nabla_\theta P(\tau; \theta) R(\tau) d\tau \\ &= \int P(\tau; \theta) \nabla_\theta \log P(\tau; \theta) R(\tau) d\tau \\ &= \mathbb{E}_{\tau \sim \pi_\theta} \left[ \nabla_\theta \log P(\tau; \theta) R(\tau) \right] \end{aligned}$$

Expanding $\log P(\tau; \theta)$:

$$\log P(\tau; \theta) = \log \mu(s_0) + \sum_{t=0}^{T-1} \log \pi_\theta(a_t \mid s_t) + \sum_{t=0}^{T-1} \log \mathcal{P}(s_{t+1} \mid s_t, a_t)$$

Taking the gradient with respect to $\theta$ eliminates dynamics terms ($\mu$ and $\mathcal{P}$):

$$\nabla_\theta \log P(\tau; \theta) = \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t \mid s_t)$$

Substituting back yields the fundamental Policy Gradient formula:

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t \mid s_t) G_t \right]$$

---

## 3. Variance Reduction via Baselines
To reduce the high variance of $G_t$, a state-dependent baseline $b(s_t)$ (typically $V(s_t)$) is subtracted:

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t \mid s_t) \left( G_t - b(s_t) \right) \right]$$

### Unbiased Baseline Proof
$$\mathbb{E}_{a_t \sim \pi_\theta} \left[ \nabla_\theta \log \pi_\theta(a_t \mid s_t) b(s_t) \right] = b(s_t) \sum_{a_t} \pi_\theta(a_t \mid s_t) \frac{\nabla_\theta \pi_\theta(a_t \mid s_t)}{\pi_\theta(a_t \mid s_t)} = b(s_t) \nabla_\theta \sum_{a_t} \pi_\theta(a_t \mid s_t) = b(s_t) \nabla_\theta (1) = 0$$

Subtraction of baseline $b(s_t)$ introduces zero bias while significantly reducing variance.