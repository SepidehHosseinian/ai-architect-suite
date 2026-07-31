# 🧮 RL Math Foundation 04: Preference Alignment (RLHF, DPO, KTO)

## 1. Bradley-Terry Preference Model (RLHF)
Human preferences over pairs of responses $(y_w \succ y_l \mid x)$ (where $y_w$ is preferred over $y_l$) are modeled using the Bradley-Terry formulation:

$$P^*(y_w \succ y_l \mid x) = \sigma \left( r^*(x, y_w) - r^*(x, y_l) \right) = \frac{1}{1 + \exp\left(-(r^*(x, y_w) - r^*(x, y_l))\right)}$$

The reward model $r_\phi(x, y)$ is trained by minimizing negative log-likelihood:

$$\mathcal{L}_R(\phi) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma \left( r_\phi(x, y_w) - r_\phi(x, y_l) \right) \right]$$

---

## 2. Direct Preference Optimization (DPO) Derivation

DPO eliminates the explicit reward model by solving for $r(x, y)$ directly in terms of policy ratio $\pi_\theta / \pi_{\text{ref}}$.

### Step 1: KL-Constrained RL Objective
$$\max_{\pi} \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi} [r(x, y)] - \beta D_{\text{KL}}(\pi(y \mid x) \parallel \pi_{\text{ref}}(y \mid x))$$

### Step 2: Closed-Form Policy Solution
Solving the unconstrained optimization problem via Lagrange multipliers yields:

$$\pi^*(y \mid x) = \frac{\pi_{\text{ref}}(y \mid x) \exp\left( \frac{1}{\beta} r(x, y) \right)}{Z(x)}$$

Where $Z(x) = \sum_y \pi_{\text{ref}}(y \mid x) \exp\left( \frac{1}{\beta} r(x, y) \right)$ is the partition function.

### Step 3: Inverting to Express Reward
Taking logs on both sides:

$$\log \pi^*(y \mid x) = \log \pi_{\text{ref}}(y \mid x) + \frac{1}{\beta} r(x, y) - \log Z(x)$$

$$r(x, y) = \beta \log \frac{\pi^*(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x)$$

### Step 4: Substituting into Bradley-Terry Loss
Substituting $r(x, y)$ into the Bradley-Terry objective cancels out partition function terms $Z(x)$:

$$r(x, y_w) - r(x, y_l) = \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}$$

This yields the exact **DPO Objective Function**:

$$\mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right) \right]$$

---

## 3. Kahneman-Tversky Optimization (KTO)
KTO removes pairwise preferences $(y_w, y_l)$, operating directly on un-paired binary signals (desirable / undesirable outputs) using prospect theory value functions:

$$\mathcal{L}_{\text{KTO}}(\theta) = \mathbb{E}_{(x, y, z)} \left[ w(z) \cdot v\left( \beta \log \frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)} - z_0 \right) \right]$$

Where $z \in \{+1, -1\}$ denotes desirable/undesirable feedback, and $v(v)$ applies asymmetrical loss aversion weighting.
5. rl_foundations/05_reasoning_rl_grpo.md
Markdown
# 🧮 RL Math Foundation 05: Modern Reasoning RL & Group Relative Policy Optimization (GRPO)

## 1. Overview & Architectural Motivation
Traditional PPO requires an explicit critic model $V_\phi(s)$ of equivalent size to the policy network $\pi_\theta$ (e.g., a 70B policy requires a 70B critic), doubling memory consumption during training. 

**Group Relative Policy Optimization (GRPO)** eliminates the value network entirely by sampling a group of $G$ candidate outputs per prompt and estimating baselines using the relative score statistics within each sampled group.

---

## 2. GRPO Mathematical Formulation

For each input prompt $q$, GRPO samples a group of $G$ distinct output responses $\{o_1, o_2, \dots, o_G\}$ from old policy $\pi_{\theta_{\text{old}}}$.

```text
[ Prompt q ] ───► Sample G Outputs ───► Evaluator / Reward Models ───► [ Rewards {r_1, r_2, ..., r_G} ]
                                                                                   │
                                                                                   ▼
                                                                     Normalized Advantage Calculation
                                                                     A_i = (r_i - Mean(r)) / Std(r)
```
## 1. Group Advantage Normalization
Reward models or deterministic evaluators score each output to yield rewards $\{r_1, r_2, \dots, r_G\}$. Normalized group advantages $A_i$ are computed as:$$\bar{r} = \frac{1}{G} \sum_{j=1}^{G} r_j, \quad \sigma_r = \sqrt{\frac{1}{G} \sum_{j=1}^{G} (r_j - \bar{r})^2 + \epsilon}$$$$A_i = \frac{r_i - \bar{r}}{\sigma_r}$$

## 2. Objective Function $J_{\text{GRPO}}(\theta)$
- $$J_{\text{GRPO}}(\theta) = \hat{\mathbb{E}}_{q \sim \mathcal{D}, \{o_i\}_{i=1}^G \sim \pi_{\theta_{\text{old}}}} \left[ \frac{1}{G} \sum_{i=1}^{G} \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \min \left( \frac{\pi_\theta(o_{i,t} \mid q, o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t} \mid q, o_{i,<t})} A_i, \; \text{clip}\left(\frac{\pi_\theta(o_{i,t} \mid q, o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t} \mid q, o_{i,<t})}, 1-\epsilon, 1+\epsilon\right) A_i \right) - \beta D_{\text{KL}}(\pi_\theta \parallel \pi_{\text{ref}}) \right]$$

## 3. Per-Token KL Penalty Estimation
To maintain policy stability without an auxiliary model, KL divergence is computed analytically per token:$$D_{\text{KL}}(\pi_\theta \parallel \pi_{\text{ref}}) = \frac{\pi_{\text{ref}}(o_{i,t} \mid q, o_{i,<t})}{\pi_\theta(o_{i,t} \mid q, o_{i,<t})} - \log \frac{\pi_{\text{ref}}(o_{i,t} \mid q, o_{i,<t})}{\pi_\theta(o_{i,t} \mid q, o_{i,<t})} - 1$$

## 4. PPO vs. GRPO Structural Comparison
FeatureStandard PPOGroup Relative Policy Optimization (GRPO)Critic Model MemoryEqual to Policy Size ($100\%$ Memory Overhead)Zero Memory Overhead (No Critic Model)Baseline EstimationState-Value Function $V_\phi(s)$Group Mean Reward ($\bar{r} = \frac{1}{G} \sum r_i$)Target TasksGeneral Text Generation & ChatMulti-step Reasoning, Code, & Mathematical ProofsSample EfficiencySingle response per update stepMulti-sample group comparisons ($G \ge 4$)                                                                   