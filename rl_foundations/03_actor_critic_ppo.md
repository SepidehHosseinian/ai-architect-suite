# 🧮 RL Math Foundation 03: Actor-Critic Architecture & PPO

## 1. Actor-Critic Framework & Advantage Estimation
Actor-Critic methods split learning into two networks:
* **Actor $\pi_\theta(a \mid s)$:** Updates policy parameters along the policy gradient.
* **Critic $V_\phi(s)$:** Evaluates state values to approximate expected return.

### Advantage Function
$$A^\pi(s_t, a_t) = Q^\pi(s_t, a_t) - V^\pi(s_t)$$

### Generalized Advantage Estimation (GAE)
Using TD errors $\delta_t^V = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$, GAE balances bias and variance via hyperparameter $\lambda \in [0, 1]$:

$$\hat{A}_t^{\text{GAE}(\gamma, \lambda)} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}^V$$

---

## 2. Proximal Policy Optimization (PPO) Clipped Objective

To prevent destructive policy updates without computationally expensive natural gradient matrix inversions (TRPO), PPO limits policy ratios:

$$r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}$$

### PPO Clipped Surrogate Objective $L^{\text{CLIP}}(\theta)$

$$L^{\text{CLIP}}(\theta) = \hat{\mathbb{E}}_t \left[ \min \left( r_t(\theta) \hat{A}_t, \; \text{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon) \hat{A}_t \right) \right]$$

Where $\epsilon$ is a hyperparameter (typically $\epsilon = 0.1$ or $0.2$).

```text
                  Advantage > 0                       Advantage < 0
          L_CLIP                              L_CLIP
            ▲                                   ▲
            │      /                            │
(1+ε)A_t ───┼─────/─── (Clipped)                │
            │    /                              │
     A_t ───┼───/                               │           /
            │  /                                │          /
            │ /                          (1-ε)A_t ───/────/────── (Clipped)
            │/                                  │   /
────────────┼────────────────► r                │  /
            │ 1.0   1+ε                         │ / 1-ε   1.0
                                    ────────────┼────────────────► r
```
## Full PPO Composite Loss Function
-$$L^{\text{PPO}}(\theta, \phi) = \hat{\mathbb{E}}_t \left[ L^{\text{CLIP}}_t(\theta) - c_1 L^{\text{VF}}_t(\phi) + c_2 S[\pi_\theta](s_t) \right]$$Where $L^{\text{VF}}_t(\phi) = (V_\phi(s_t) - V_t^{\text{target}})^2$ is squared-error value loss, and $S[\pi_\theta](s_t)$ is policy entropy for exploration encouragement.